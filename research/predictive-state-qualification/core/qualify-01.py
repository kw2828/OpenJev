"""One original fabricated-only core qualification; no empirical inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/predictive-state-core-engineering-v1/qualification-01'
SOURCES = ('src/openjev/research/predictive_state_correction.py', 'tests/test_predictive_state_correction.py')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COMMANDS = (('.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', SOURCES[1]),
            ('.venv/bin/ruff', 'check', *SOURCES))


def pin(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    initial = {name: pin(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        dest = OUT / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        assert pin(dest) == initial[name]
    env = os.environ.copy()
    env.update({key: '1' for key in THREADS})
    env['PYTHONPATH'] = str(ROOT / 'src')
    save('preflight.json', {'sources': initial, 'helper': pin(Path(__file__)),
          'commands': COMMANDS, 'thread_env': {key: env[key] for key in THREADS},
          'scope': 'Fabricated mathematical and causal fixtures only. No dataset reads, empirical fits, microbenchmarks or scientific outcomes.',
          'per_command_timeout_seconds': 180})
    records = []
    for number, command in enumerate(COMMANDS, 1):
        started = time.perf_counter()
        log = OUT / f'command-{number:02d}.log'
        with log.open('wb') as handle:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, timeout=180, check=False)
                code, state = result.returncode, 'EXITED'
            except subprocess.TimeoutExpired:
                code, state = None, 'TIMEOUT'
        record = {'command': command, 'state': state, 'returncode': code,
                  'elapsed_seconds': time.perf_counter() - started,
                  'log': str(log.relative_to(ROOT)), 'log_pin': pin(log)}
        save(f'command-{number:02d}.json', record)
        records.append(record)
        if state != 'EXITED' or code != 0:
            break
    final = {name: pin(ROOT / name) for name in SOURCES}
    unchanged = final == initial
    passed = len(records) == len(COMMANDS) and all(row['state'] == 'EXITED' and row['returncode'] == 0 for row in records) and unchanged
    receipt = {'status': 'PASS' if passed else 'FAILED', 'sources_before': initial,
               'sources_after': final, 'sources_unchanged': unchanged, 'commands': records,
               'scope': 'Fabricated-only component qualification. No empirical data or performance evidence.'}
    save('receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'sources_unchanged': unchanged,
                      'commands': records}, indent=2))
    for row in records:
        print((ROOT / row['log']).read_text())
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
