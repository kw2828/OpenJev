import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path.cwd()
out = Path(__file__).resolve().parent
sources = ('scripts/audit_fsm_author_nllfr.py', 'tests/test_audit_fsm_author_nllfr.py')
commands = [['.venv/bin/ruff', 'check', *sources], ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', sources[1]]]
keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
env = {**os.environ, **dict.fromkeys(keys, '1'), 'PYTHONDONTWRITEBYTECODE': '1'}

def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def save(name, value):
    with (out/name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

before = {n: pin(root/n) for n in sources}
for name in sources:
    ast.parse((root/name).read_text())
    target = out/'source'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/name, target)
preflight = {'sources': before, 'snapshots': {n: pin(out/'source'/n) for n in sources},
             'commands': commands, 'environment': {k: env[k] for k in (*keys, 'PYTHONDONTWRITEBYTECODE')},
             'python': sys.version, 'executable': sys.executable, 'wrapper': pin(__file__),
             'scope': 'Fabricated arrays and opaque fake evidence only. No study admission, measured data, author package, fitting, or empirical replay.'}
save('preflight.json', preflight)
rows, error, began = [], None, time.perf_counter()
try:
    for number, command in enumerate(commands, 1):
        path = out/f'command-{number:02d}.log'
        start = time.perf_counter()
        with path.open('xb') as handle:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
                error = 'Qualification command exceeded120seconds'
        rows.append({'command': command, 'pid': child.pid, 'returncode': code, 'seconds': time.perf_counter()-start, 'log': pin(path)})
        if code != 0 or error:
            break
except BaseException as exc:
    error = f'{type(exc).__name__}: {exc}'
after = {n: pin(root/n) for n in sources}
passed = error is None and len(rows) == len(commands) and all(r['returncode'] == 0 for r in rows) and before == after
result = {'status': 'PASS' if passed else 'FAIL', 'error': error, 'commands': rows, 'seconds': time.perf_counter()-began,
          'sources_before': before, 'sources_after': after, 'sources_unchanged': before == after,
          'preflight': pin(out/'preflight.json'), 'scope': preflight['scope']}
save('receipt.json', result)
print(json.dumps({'status': result['status'], 'receipt': pin(out/'receipt.json'), 'commands': rows}, indent=2))
raise SystemExit(0 if passed else 1)
