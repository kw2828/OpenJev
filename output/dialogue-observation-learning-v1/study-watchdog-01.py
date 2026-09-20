"""Local process supervision; no experiment or metric logic."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

prefix, *command = sys.argv[1:]
prefix = Path(prefix)
started = time.monotonic()
log_path = prefix.with_suffix('.log')
launch_path = prefix.with_suffix('.launch.json')
terminal_path = prefix.with_suffix('.terminal.json')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def exists_group(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


process = None
error = None
timed_out = False
try:
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=Path.cwd(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(launch_path, {'command': command, 'pid': process.pid, 'pgid': process.pid,
            'started_unix': time.time(), 'cap_seconds': 28800,
            'watchdog_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        try:
            process.wait(timeout=max(0, 28800 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
except BaseException as caught:
    error = repr(caught)
finally:
    if process is not None:
        if exists_group(process.pid):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if exists_group(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        terminal = {'returncode': process.returncode, 'pgid': process.pid,
                    'group_absent': not exists_group(process.pid)}
    else:
        terminal = {'returncode': None, 'pgid': None, 'group_absent': True}
    terminal.update({'command': command, 'timed_out': timed_out, 'error': error,
                     'wall_seconds': time.monotonic() - started, 'finished_unix': time.time()})
    write(terminal_path, terminal)
    print(json.dumps(terminal), flush=True)
if timed_out or error or terminal['returncode'] != 0 or not terminal['group_absent']:
    raise SystemExit(1)
