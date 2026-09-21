"""Capture one bounded source-only qualification, with no scientific data."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = (
    'src/openjev/research/otto_symmetry_head.py',
    'scripts/study_otto_symmetry_head.py',
    'scripts/audit_otto_symmetry_head.py',
    'tests/test_otto_symmetry_head.py',
    'tests/test_otto_symmetry_study.py',
    'tests/test_audit_otto_symmetry_head.py',
    'research/otto-symmetry-head-protocol.md',
)


def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}


def main():
    output = Path(__file__).parent/'combined-preflight-01'
    output.mkdir(exist_ok=False)
    before = pins()
    environment = {**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE': '1',
                   **{name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    commands = [
        ('pytest', [sys.executable, '-m', 'pytest', '-q', *FILES[3:6]]),
        ('ruff', [sys.executable, '-m', 'ruff', 'check', *FILES[:6]]),
        ('diff', ['git', 'diff', '--check']),
    ]
    results = []
    for label, command in commands:
        start = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=60)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
            error = None
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = exc.stdout or b'', exc.stderr or b'', None
            error = repr(exc)
        (output/f'{label}.stdout').write_bytes(stdout)
        (output/f'{label}.stderr').write_bytes(stderr)
        results.append({'name': label, 'command': command, 'returncode': code, 'error': error,
                        'seconds': time.perf_counter()-start})
        print(label, code, stdout.decode(errors='replace'), stderr.decode(errors='replace'), flush=True)
        if code != 0:
            break
    after = pins()
    complete = len(results) == len(commands) and all(r['returncode'] == 0 for r in results) and before == after
    receipt = {'status': 'completed' if complete else 'failed', 'sources_before': before, 'sources_after': after,
               'commands': results, 'python_executable': sys.executable, 'scientific_training_calls': 0,
               'native_simulator_calls': 0, 'scope': 'Synthetic unit fixtures only; no scientific efficacy evidence.',
               'files': {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                         for p in sorted(output.iterdir())}}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
