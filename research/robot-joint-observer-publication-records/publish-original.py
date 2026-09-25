"""Original bounded publication of independently audited saved evidence."""
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
BASE = ROOT/'output/robot-joint-observer-publication-engineering-v1'
SOURCE = 'scripts/publish_robot_joint_observer.py'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')

def descriptor(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

closure = ROOT/'output/robot-joint-observer-engineering-v1/audit-closure-01.json'
assert json.loads(closure.read_text())['status'] == 'PASS'
latest = sorted(BASE.glob('qualification-*'))[-1]/'receipt.json'
qualification = json.loads(latest.read_text())
assert qualification['status'] == 'PASS'
assert all(descriptor(ROOT/name) == pin for name, pin in qualification['sources'].items())
command = ['.venv/bin/python', SOURCE]
launch = {'command': command, 'started_utc': datetime.now(UTC).isoformat(), 'cap_seconds': 180,
          'publisher': descriptor(ROOT/SOURCE), 'qualification': descriptor(latest),
          'audit_closure': descriptor(closure), 'thread_env': dict.fromkeys(THREADS, '1')}
write(BASE/'publication-launch-01.json', launch)
env = dict(os.environ)
env.update(launch['thread_env'])
env['PYTHONPATH'] = 'src'
started = time.monotonic()
timeout = False
with (BASE/'publication-process-01.log').open('xb') as handle:
    try:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                              stderr=subprocess.STDOUT, timeout=180, check=False).returncode
    except subprocess.TimeoutExpired:
        code = 124
        timeout = True
receipt = {**launch, 'returncode': code, 'external_timeout': timeout, 'elapsed_seconds': time.monotonic()-started,
           'log': descriptor(BASE/'publication-process-01.log')}
write(BASE/'publication-process-01.json', receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
