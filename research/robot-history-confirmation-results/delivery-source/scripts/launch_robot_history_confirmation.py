"""Launch one frozen confirmation evaluation with an original terminal receipt."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINEERING = ROOT / 'output/robot-history-confirmation-engineering-v1'
REGISTRATION = ROOT / 'research/robot-history-confirmation-registration.json'
OUTPUT = ROOT / 'output/robot-history-confirmation-v1'
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_history_confirmation.py',
           '--registration', 'research/robot-history-confirmation-registration.json',
           '--output', 'output/robot-history-confirmation-v1']
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def descriptor(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Regular nonsymlink file required: ' + str(path))
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, expected in plan['sources'].items():
        if descriptor(ROOT / name) != expected:
            raise ValueError('Source changed before launch: ' + name)
    if descriptor(__file__) != plan['launcher']:
        raise ValueError('Launcher changed before launch')
    q = plan['qualification']
    if descriptor(q['path']) != {k: q[k] for k in ('sha256', 'bytes')}:
        raise ValueError('Qualification changed')
    qualification = json.loads(Path(q['path']).read_text())
    if (qualification['status'] != 'PASS' or qualification['sources'] != plan['sources']
            or qualification['sources_unchanged'] is not True
            or qualification['thread_env'] != dict.fromkeys(THREADS, '1')):
        raise ValueError('Exact source qualification required')
    if qualification['launcher'] != plan['launcher']:
        raise ValueError('Qualified launcher required')
    for row in qualification['commands']:
        if row['returncode'] != 0 or descriptor(row['log'])['sha256'] != row['sha256']:
            raise ValueError('Qualification process or original log changed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name in (*plan['sources'], str(REGISTRATION.relative_to(ROOT)),
                 'scripts/launch_robot_history_confirmation.py'):
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Source not frozen in pre-access commit: ' + name)
    if plan['config']['wall_cap_seconds'] != 900.:
        raise ValueError('Fixed 900-second native and 960-second external caps required')
    if OUTPUT.exists():
        raise ValueError('Study output already exists; no restart')
    env = dict(os.environ)
    env.update(dict.fromkeys(THREADS, '1'))
    record = {'command': COMMAND, 'pre_access_commit': commit,
              'started_utc': datetime.now(UTC).isoformat(),
              'registration_sha256': descriptor(REGISTRATION)['sha256'],
              'launcher': descriptor(__file__),
              'thread_env': {key: env[key] for key in THREADS},
              'scope': 'Twenty-four fixed checkpoints and four references; two reserved confirmation recordings; no training, selection, restart or official TEST access'}
    write(ENGINEERING / 'run-launch-01.json', record)
    started, timeout = time.monotonic(), False
    with (ENGINEERING / 'run-process-01.log').open('xb') as log:
        try:
            process = subprocess.run(COMMAND, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, check=False,
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
