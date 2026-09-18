"""One-shot detached lifecycle for the frozen pin-quality recovery study."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
WORKER = ROOT / 'scripts/chess_pin_quality_recovery.py'
CAPS = {'run': 43200, 'audit': 7200}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def source_key(path, root):
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def validate_intent(intent):
    root, plan_path, worker = (Path(intent[k]) for k in ('root', 'plan', 'worker'))
    if digest(plan_path) != intent['plan_sha256']:
        raise ValueError('Frozen plan changed')
    plan = read(plan_path)
    for path, key in ((SELF, 'launcher_sha256'), (worker, 'worker_sha256')):
        actual = digest(path)
        if actual != intent[key] or plan['sources'].get(source_key(path, root)) != actual:
            raise ValueError('Launcher or recovery source differs from frozen plan')
    expected_caps = {'run': plan['protocol']['time_cap_seconds'],
                     'audit': plan['protocol']['audit_time_cap_seconds']}
    if intent['caps'] != expected_caps:
        raise ValueError('Watchdog ceilings differ from frozen plan')
    for phase, value in expected_caps.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid watchdog ceiling')
        if not 0 < value <= CAPS[phase]:
            raise ValueError('Watchdog exceeds fixed maximum ceiling')
    if Path(intent['python']) != Path(sys.executable).absolute():
        raise ValueError('Interpreter identity changed')
    return plan


def launch(plan, out, execution, audit, *, root=ROOT, worker=WORKER):
    """Return the detached supervisor handle, never retry or reuse an output tree."""
    root = Path(root).resolve()
    paths = [Path(p).absolute() for p in (plan, out, execution, audit, worker)]
    plan, out, execution, audit, worker = paths
    for path in paths:
        if path.resolve() != path:
            raise ValueError('Symlink or noncanonical lifecycle path')
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate((out, execution, audit)) for b in (out, execution, audit)[i+1:]):
        raise ValueError('Lifecycle outputs must be disjoint')
    if execution.exists() or audit.exists():
        raise FileExistsError('Execution or audit already exists; no retry or resume')
    p = read(plan)
    intent = {'status': 'launch_intent', 'created_unix': time.time(), 'root': str(root),
              'plan': str(plan), 'out': str(out), 'execution': str(execution), 'audit': str(audit),
              'worker': str(worker), 'python': str(Path(sys.executable).absolute()),
              'plan_sha256': digest(plan), 'launcher_sha256': digest(SELF), 'worker_sha256': digest(worker),
              'caps': {'run': p['protocol']['time_cap_seconds'], 'audit': p['protocol']['audit_time_cap_seconds']},
              'retry_allowed': False, 'resume_allowed': False}
    validate_intent(intent)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'intent.json', intent)
    command = [sys.executable, str(SELF), 'supervise', '--out', str(out)]
    begin = time.time()
    try:
        with (out / 'supervisor.log').open('x') as log:
            process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    except BaseException as error:
        write(out / 'launch-failed.json', {'status': 'spawn_failed', 'command': command,
              'started_unix': begin, 'finished_unix': time.time(), 'error': repr(error),
              'plan_sha256': intent['plan_sha256']})
        raise
    write(out / 'launch.json', {'status': 'launched', 'command': command, 'pid': process.pid,
          'started_unix': begin, 'observed_unix': time.time(), 'start_new_session': True,
          'plan_sha256': intent['plan_sha256'], 'launcher_sha256': intent['launcher_sha256'],
          'worker_sha256': intent['worker_sha256'], 'intent_sha256': digest(out / 'intent.json'),
          'observation_not_completion': True})
    return process


def stop_owned_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_phase(intent, phase):
    out = Path(intent['out'])
    command = [intent['python'], intent['worker'], phase, '--plan', intent['plan'],
               '--out', intent['execution'] if phase == 'run' else intent['audit']]
    if phase == 'audit':
        command += ['--execution', intent['execution']]
    begin, clock = time.time(), time.monotonic()
    process = None
    status, error = 'spawn_failed', None
    try:
        validate_intent(intent)
        with (out / f'{phase}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=intent['root'], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            write(out / f'{phase}-started.json', {'command': command, 'pid': process.pid,
                  'started_unix': begin, 'plan_sha256': intent['plan_sha256'],
                  'watchdog_seconds': intent['caps'][phase]})
            try:
                process.wait(timeout=max(0, intent['caps'][phase] - (time.monotonic() - clock)))
                status = 'exited'
            except subprocess.TimeoutExpired:
                status = 'watchdog_terminated'
                stop_owned_worker(process)
    except BaseException as caught:  # noqa: BLE001 - Record interruptions and stop only our worker.
        error = repr(caught)
        if process is not None:
            status = 'supervisor_error_terminated_worker'
            stop_owned_worker(process)
    terminal = {'phase': phase, 'status': status, 'command': command,
                'pid': process.pid if process else None, 'started_unix': begin,
                'finished_unix': time.time(), 'wall_seconds': time.monotonic() - clock,
                'returncode': process.returncode if process else None,
                'watchdog_seconds': intent['caps'][phase], 'plan_sha256': intent['plan_sha256'],
                'error': error, 'raw_worker_receipts_modified': False}
    write(out / f'{phase}-terminal.json', terminal)
    return status == 'exited' and terminal['returncode'] == 0


def validate_completed(execution, plan_sha256):
    """Require exact regular-file membership and hashes before allowing audit."""
    execution = Path(execution)
    receipt = execution / 'completed.json'
    if receipt.is_symlink() or not receipt.is_file() or (execution / 'failed.json').exists():
        raise ValueError('Missing or conflicting run completion receipt')
    saved = read(receipt)
    if saved.get('status') != 'completed' or saved.get('plan_sha256') != plan_sha256:
        raise ValueError('Run completion identity differs')
    files = saved.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty or invalid run manifest')
    actual = {}
    for path in execution.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in execution artifacts')
        if path.is_file() and path != receipt:
            actual[str(path.relative_to(execution))] = digest(path)
        elif not path.is_dir() and path != receipt:
            raise ValueError('Nonregular execution artifact')
    if files != actual:
        raise ValueError('Run manifest membership or hashes differ')
    return digest(receipt)


def supervise(out):
    out = Path(out).absolute()
    intent = read(out / 'intent.json')
    # A second supervisor must fail before it can start a worker.
    write(out / 'supervisor-started.json', {'pid': os.getpid(), 'ppid': os.getppid(),
          'session_id': os.getsid(0), 'started_unix': time.time(), 'intent_sha256': digest(out / 'intent.json')})
    begin = time.time()
    status, error, completed_hash = 'supervisor_failed', None, None
    try:
        if Path(intent['out']) != out:
            raise ValueError('Intent launcher output differs')
        validate_intent(intent)
        if Path(intent['execution']).exists() or Path(intent['audit']).exists():
            raise FileExistsError('Execution or audit exists before worker launch')
        if not run_phase(intent, 'run'):
            status = 'run_failed'
        else:
            status = 'invalid_run_receipt'
            completed_hash = validate_completed(intent['execution'], intent['plan_sha256'])
            validate_intent(intent)
            if not run_phase(intent, 'audit'):
                status = 'audit_failed'
            else:
                status = 'invalid_audit_receipt'
                audit = Path(intent['audit'])
                receipt = audit / 'receipt.json'
                if receipt.is_symlink() or not receipt.is_file() or (audit / 'failed.json').exists():
                    raise ValueError('Missing or conflicting audit receipt')
                saved = read(receipt)
                if (saved.get('status') != 'completed' or saved.get('plan_sha256') != intent['plan_sha256']
                        or saved.get('execution_receipt_sha256') != completed_hash):
                    raise ValueError('Audit completion identity differs')
                validate_intent(intent)
                if validate_completed(intent['execution'], intent['plan_sha256']) != completed_hash:
                    raise ValueError('Execution changed during audit')
                status = 'completed'
    except BaseException as caught:  # noqa: BLE001 - Preserve a terminal lifecycle receipt on interruption.
        error = repr(caught)
    returncode = 0 if status == 'completed' else 1
    write(out / 'terminal.json', {'status': status, 'pid': os.getpid(), 'started_unix': begin,
          'command': [sys.executable, str(SELF), 'supervise', '--out', str(out)], 'returncode': returncode,
          'finished_unix': time.time(), 'plan_sha256': intent['plan_sha256'], 'error': error,
          'execution_receipt_sha256': completed_hash, 'retry_performed': False,
          'scope': 'Process lifecycle and receipt integrity only; scientific acceptance is in the audited study.'})
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'supervise'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.command == 'supervise':
        return supervise(args.out)
    if any(x is None for x in (args.plan, args.execution, args.audit)):
        parser.error('launch requires --plan, --execution and --audit')
    process = launch(args.plan, args.out, args.execution, args.audit)
    print(json.dumps({'status': 'launched', 'pid': process.pid, 'launcher': str(args.out.absolute())}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
