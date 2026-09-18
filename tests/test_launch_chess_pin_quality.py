"""Synthetic process lifecycle tests; no chess data, models or engine calls."""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/launch_chess_pin_quality.py'
spec = importlib.util.spec_from_file_location('pin_launcher_test', SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

FAKE_WORKER = textwrap.dedent('''
    import argparse, hashlib, json, os, pathlib, sys, time
    p=argparse.ArgumentParser()
    p.add_argument('phase'); p.add_argument('--plan'); p.add_argument('--out'); p.add_argument('--execution')
    a=p.parse_args(); plan=pathlib.Path(a.plan); out=pathlib.Path(a.out)
    data=json.loads(plan.read_text()); mode=data['fixture']
    h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    write=lambda n,v: (out/n).write_text(json.dumps(v))
    out.mkdir(parents=True,exist_ok=False)
    write('started.json',{'pid':os.getpid(),'plan_sha256':h(plan)})
    print('fixture phase '+a.phase,flush=True)
    if mode=='parent_exit' and a.phase=='run':
        while not (plan.parent/'parent_returned.txt').exists(): time.sleep(.02)
    if mode == a.phase+'_timeout': time.sleep(20)
    if mode == a.phase+'_failure':
        write('failed.json',{'status':'failed','fixture':True})
        sys.exit(7)
    if a.phase=='run':
        time.sleep(.05)
        write('summary.json',{'status':'completed','fixture':True})
        files={str(f.relative_to(out)):h(f) for f in out.rglob('*') if f.is_file()}
        write('completed.json',{'status':'completed','plan_sha256':'wrong' if mode=='bad_run_plan' else h(plan),
                                'files':files})
        if mode=='bad_run_files': write('summary.json',{'tampered':True})
    else:
        write('receipt.json',{'status':'completed','plan_sha256':'wrong' if mode=='bad_audit' else h(plan),
                              'execution_receipt_sha256':h(pathlib.Path(a.execution)/'completed.json')})
''')


def fixture_paths(tmp_path, mode='success'):
    root = tmp_path.resolve()
    worker = root / 'fixture_worker.py'
    worker.write_text(FAKE_WORKER)
    plan = root / 'plan.json'
    protocol = {'time_cap_seconds': .15 if mode == 'run_timeout' else 10,
                'audit_time_cap_seconds': .15 if mode == 'audit_timeout' else 10}
    launcher.write(plan, {'protocol': protocol, 'fixture': mode,
                         'sources': {str(SCRIPT): launcher.digest(SCRIPT),
                                     'fixture_worker.py': launcher.digest(worker)}})
    return {'root': root, 'worker': worker, 'plan': plan, 'out': root / 'launcher',
            'execution': root / 'execution', 'audit': root / 'audit'}


def finish(process, directory):
    try:
        return process.wait(timeout=15)
    finally:
        if process.poll() is None:
            # Stop only this fixture's recorded child and supervisor if a test fails.
            for path in directory.glob('*-started.json'):
                if path.name == 'supervisor-started.json':
                    continue
                pid = launcher.read(path)['pid']
                try:
                    os.kill(pid, 15)
                except ProcessLookupError:
                    pass
            process.terminate()
            process.wait(timeout=5)


def test_detached_success_retains_exact_commands_and_receipts(tmp_path):
    paths = fixture_paths(tmp_path)
    process = launcher.launch(**paths)
    assert os.getsid(process.pid) == process.pid
    assert finish(process, paths['out']) == 0
    intent = launcher.read(paths['out'] / 'intent.json')
    launched = launcher.read(paths['out'] / 'launch.json')
    done = launcher.read(paths['out'] / 'terminal.json')
    assert launched['pid'] == process.pid and launched['start_new_session']
    assert done['status'] == 'completed' and not done['retry_performed']
    assert launched['plan_sha256'] == launcher.digest(paths['plan'])
    assert launched['launcher_sha256'] == launcher.digest(SCRIPT)
    assert launched['worker_sha256'] == launcher.digest(paths['worker'])
    for phase in ('run', 'audit'):
        receipt = launcher.read(paths['out'] / f'{phase}-terminal.json')
        started = launcher.read(paths['out'] / f'{phase}-started.json')
        expected = [sys.executable, str(paths['worker']), phase, '--plan', str(paths['plan']),
                    '--out', str(paths['execution'] if phase == 'run' else paths['audit'])]
        if phase == 'audit':
            expected += ['--execution', str(paths['execution'])]
        assert receipt['command'] == expected
        assert started['pid'] == receipt['pid']
        assert receipt['status'] == 'exited' and receipt['returncode'] == 0
        assert receipt['finished_unix'] >= receipt['started_unix']
        assert receipt['watchdog_seconds'] == intent['caps'][phase]
        assert not receipt['raw_worker_receipts_modified']
    for log in paths['out'].glob('*.log'):
        assert stat.S_ISREG(log.stat().st_mode) and not log.is_symlink()


@pytest.mark.parametrize('mode,expected,phase_status,audit_exists', [
    ('run_failure', 'run_failed', 'exited', False),
    ('bad_run_plan', 'invalid_run_receipt', 'exited', False),
    ('bad_run_files', 'invalid_run_receipt', 'exited', False),
    ('run_timeout', 'run_failed', 'watchdog_terminated', False),
    ('audit_failure', 'audit_failed', 'exited', True),
    ('audit_timeout', 'audit_failed', 'watchdog_terminated', True),
    ('bad_audit', 'invalid_audit_receipt', 'exited', True),
])
def test_failure_never_retries_or_audits_invalid_run(tmp_path, mode, expected, phase_status, audit_exists):
    paths = fixture_paths(tmp_path, mode)
    process = launcher.launch(**paths)
    assert finish(process, paths['out']) == 1
    done = launcher.read(paths['out'] / 'terminal.json')
    assert done['status'] == expected and not done['retry_performed']
    assert paths['audit'].exists() == audit_exists
    phase = 'audit' if mode.startswith('audit') else 'run'
    terminal = launcher.read(paths['out'] / f'{phase}-terminal.json')
    assert terminal['status'] == phase_status
    if mode.endswith('_failure'):
        assert terminal['returncode'] == 7
        raw = launcher.read(paths['execution' if phase == 'run' else 'audit'] / 'failed.json')
        assert raw == {'status': 'failed', 'fixture': True}
    if mode.endswith('_timeout'):
        assert terminal['returncode'] < 0
        assert not (paths['execution' if phase == 'run' else 'audit'] / 'failed.json').exists()
    assert len(list(paths['out'].glob(f'{phase}-started.json'))) == 1


@pytest.mark.parametrize('existing', ('out', 'execution', 'audit'))
def test_exclusive_output_paths_reject_reuse(tmp_path, existing):
    paths = fixture_paths(tmp_path)
    paths[existing].mkdir()
    marker = paths[existing] / 'unchanged.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        launcher.launch(**paths)
    assert marker.read_text() == 'preserve'
    assert not (paths['out'] / 'launch.json').exists()


def test_changed_frozen_source_rejected_before_detaching(tmp_path):
    paths = fixture_paths(tmp_path)
    paths['worker'].write_text(FAKE_WORKER + '\n# modified\n')
    with pytest.raises(ValueError, match='source differs'):
        launcher.launch(**paths)
    assert not paths['out'].exists()


def test_disjoint_paths_and_budget_caps(tmp_path):
    paths = fixture_paths(tmp_path)
    with pytest.raises(ValueError, match='disjoint'):
        launcher.launch(**{**paths, 'audit': paths['execution'] / 'audit'})
    plan = launcher.read(paths['plan'])
    plan['protocol']['time_cap_seconds'] = 43201
    paths['plan'].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='ceiling'):
        launcher.launch(**paths)
    assert not paths['out'].exists()


def test_duplicate_supervisor_cannot_start_second_worker(tmp_path):
    paths = fixture_paths(tmp_path)
    process = launcher.launch(**paths)
    assert finish(process, paths['out']) == 0
    before = {p.name: launcher.digest(p) for p in paths['out'].iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        launcher.supervise(paths['out'])
    assert before == {p.name: launcher.digest(p) for p in paths['out'].iterdir() if p.is_file()}


def test_supervisor_survives_launching_parent_exit(tmp_path):
    paths = fixture_paths(tmp_path, 'parent_exit')
    program = '''
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('fixture_launcher',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
process=module.launch(**json.loads(sys.argv[2]))
print(process.pid,flush=True)
'''
    parent = subprocess.run([sys.executable, '-c', program, str(SCRIPT),
                             json.dumps({k: str(v) for k, v in paths.items()})],
                            capture_output=True, text=True, timeout=5, check=True)
    pid = int(parent.stdout.strip())
    try:
        # The parent exited while its detached worker waits for our release.
        assert os.getsid(pid) == pid
        assert not (paths['out'] / 'terminal.json').exists()
        (paths['root'] / 'parent_returned.txt').write_text('parent process already exited')
        end = time.monotonic() + 10
        while not (paths['out'] / 'terminal.json').exists() and time.monotonic() < end:
            time.sleep(.02)
        assert launcher.read(paths['out'] / 'terminal.json')['status'] == 'completed'
    finally:
        if not (paths['out'] / 'terminal.json').exists():
            for path in paths['out'].glob('*-started.json'):
                if path.name != 'supervisor-started.json':
                    try:
                        os.kill(launcher.read(path)['pid'], 15)
                    except ProcessLookupError:
                        pass
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
