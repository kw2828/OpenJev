"""One frozen structured-transition campaign with an external terminal receipt."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / 'output/robot-structured-engineering-v1'
REGISTRATION = ROOT / 'research/robot-structured-registration.json'
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_structured_study.py',
           '--registration', 'research/robot-structured-registration.json',
           '--output', 'output/robot-structured-study-v1']
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def descriptor(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Regular file required: ' + str(path))
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, pin in plan['sources'].items():
        if descriptor(ROOT / name) != pin:
            raise ValueError('Source changed before launch: ' + name)
    if descriptor(__file__) != plan['launcher']:
        raise ValueError('Launcher changed before launch')
    q = plan['qualification']
    if descriptor(q['path']) != {k: q[k] for k in ('sha256', 'bytes')}:
        raise ValueError('Qualification changed')
    qualification = json.loads(Path(q['path']).read_text())
    if qualification['status'] != 'PASS' or qualification['sources'] != plan['sources']:
        raise ValueError('Source qualification missing')
    for row in qualification['commands']:
        if row['returncode'] != 0 or descriptor(row['log'])['sha256'] != row['sha256']:
            raise ValueError('Qualification process did not pass or log changed')
    for name, pin in plan['diagnostic_evidence'].items():
        if descriptor(pin['path']) != {k: pin[k] for k in ('sha256', 'bytes')}:
            raise ValueError('Diagnostic evidence changed: ' + name)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name in (*plan['sources'], str(REGISTRATION.relative_to(ROOT)),
                 'scripts/launch_robot_structured.py'):
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        if committed != (ROOT / name).read_bytes():
            raise ValueError('Source not frozen in prefit commit: ' + name)
    if (ROOT / 'output/robot-structured-study-v1').exists():
        raise ValueError('Study already exists; no restart')
    env = dict(os.environ)
    env.update({key: '1' for key in THREADS})
    record = {'command': COMMAND, 'prefit_commit': commit,
              'started_utc': datetime.now(UTC).isoformat(),
              'registration_sha256': descriptor(REGISTRATION)['sha256'],
              'launcher': descriptor(__file__), 'thread_env': {key: env[key] for key in THREADS},
              'scope': 'One registered development campaign; 30 fresh attempts, 6 cached fits; no restart; closed CONFIRM/TEST'}
    write(ENGINEERING / 'run-launch-01.json', record)
    started = time.monotonic()
    timeout = False
    with (ENGINEERING / 'run-process-01.log').open('xb') as log:
        try:
            process = subprocess.run(COMMAND, cwd=ROOT, env=env, stdout=log, check=False,
                                     stderr=subprocess.STDOUT,
                                     timeout=plan['config']['wall_cap_seconds'] + 60)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code, timeout = 124, True
    record.update(returncode=code, elapsed_seconds=time.monotonic() - started,
                  external_timeout=timeout, log=descriptor(ENGINEERING / 'run-process-01.log'))
    write(ENGINEERING / 'run-process-01.json', record)
    print(json.dumps(record), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
