"""Retain one fabricated-only source qualification attempt before registration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import UTC, datetime

ROOT = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
os.environ.update(dict.fromkeys(THREADS, '1'))
sys.path.insert(0, str(ROOT/'scripts'))
import robot_position_observer_study as study

def descriptor(path):
    raw = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')

attempt = int(sys.argv[1])
folder = ROOT/f'output/robot-position-observer-engineering-v1/qualification-{attempt:02d}'
folder.mkdir(parents=True, exist_ok=False)
names = (*study.SOURCES, study.LAUNCHER)
before = {name: descriptor(ROOT/name) for name in names}
for name in names:
    destination = folder/'sources'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT/name).read_bytes())
(folder/'qualification-helper.py').write_bytes(Path(__file__).read_bytes())
commands = [
    ['.venv/bin/ruff', 'check', *[p for p in study.QUALIFICATION_SOURCES if p.endswith('.py')]],
    ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_position_observer.py',
     'tests/test_robot_position_observer_study.py', 'tests/test_audit_robot_position_observer.py'],
]
record = {'created_utc': datetime.now(UTC).isoformat(), 'sources': {name: before[name] for name in study.SOURCES},
          'launcher': before[study.LAUNCHER], 'thread_env': dict.fromkeys(THREADS, '1'), 'commands': [],
          'scope': 'Fabricated qualification only; no empirical arrays or trained models', 'command_cap_seconds': 180}
write(folder/'launch.json', record)
for index, command in enumerate(commands, 1):
    log = folder/f'command-{index:02d}.log'
    started = time.monotonic()
    with log.open('xb') as output:
        try:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ), stdout=output,
                                    stderr=subprocess.STDOUT, check=False, timeout=180)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = {'command': command, 'returncode': code, 'seconds': time.monotonic()-started,
           'log': str(log), 'sha256': descriptor(log)['sha256']}
    record['commands'].append(row)
    write(folder/f'process-{index:02d}.json', row)
    print(json.dumps(row), flush=True)
after = {name: descriptor(ROOT/name) for name in names}
record['sources_unchanged'] = before == after
record['status'] = 'PASS' if before == after and all(row['returncode'] == 0 for row in record['commands']) else 'FAILED'
write(folder/'receipt.json', record)
print(json.dumps({'status': record['status'], 'receipt': str(folder/'receipt.json')}), flush=True)
raise SystemExit(0 if record['status'] == 'PASS' else 1)
