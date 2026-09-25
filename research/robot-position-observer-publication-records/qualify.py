"""Preserve one bounded qualification attempt for the publication helper."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
SOURCES = ('scripts/publish_robot_position_observer.py', 'tests/test_publish_robot_position_observer.py')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
BASE = ROOT/'output/robot-position-observer-publication-engineering-v1'

def descriptor(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

run = json.loads((ROOT/'output/robot-position-observer-engineering-v1/run-process-01.json').read_text())
assert run['returncode'] == 0 and run['external_timeout'] is False
attempts = sorted(BASE.glob('qualification-*'))
assert [p.name for p in attempts] == [f'qualification-{i:02d}' for i in range(1, len(attempts)+1)]
folder = BASE/f'qualification-{len(attempts)+1:02d}'
folder.mkdir()
sources = {name: descriptor(ROOT/name) for name in SOURCES}
commands = [['.venv/bin/ruff', 'check', *SOURCES],
            ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', SOURCES[1]]]
preflight = {'sources': sources, 'commands': commands, 'started_utc': datetime.now(UTC).isoformat(),
             'command_cap_seconds': 180, 'thread_env': dict.fromkeys(THREADS, '1')}
write(folder/'preflight.json', preflight)
for name in SOURCES:
    target = folder/'sources'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/name, target)
env = dict(os.environ)
env.update(preflight['thread_env'])
env['PYTHONPATH'] = 'src'
rows = []
for i, command in enumerate(commands, 1):
    log = folder/f'command-{i:02d}.log'
    started = time.monotonic()
    with log.open('xb') as handle:
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=180, check=False).returncode
        except subprocess.TimeoutExpired:
            code = 124
    rows.append({'command': command, 'returncode': code, 'seconds': time.monotonic()-started,
                 'log': str(log), 'sha256': descriptor(log)['sha256']})
    if code != 0:
        break
unchanged = sources == {name: descriptor(ROOT/name) for name in SOURCES}
receipt = {'sources': sources, 'sources_unchanged': unchanged, 'thread_env': preflight['thread_env'],
           'commands': rows, 'status': 'PASS' if unchanged and len(rows) == 2 and all(r['returncode'] == 0 for r in rows) else 'FAIL'}
write(folder/'receipt.json', receipt)
print(json.dumps({'folder': str(folder), **receipt}), flush=True)
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
