"""Preserve one fabricated-only qualification, including unsuccessful attempts."""
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'output/fsm-shift-engineering-v1/inference-qualification-01'
SOURCES = ('src/openjev/research/fsm_shift_inference.py',
           'scripts/fsm_shift_audit_math.py', 'tests/test_fsm_shift_inference.py',
           'tests/test_fsm_shift_audit_math.py')
PRIOR = ('src/openjev/__init__.py', 'src/openjev/research/__init__.py',
         'src/openjev/research/fsm_data.py', 'src/openjev/research/fsm_linear.py',
         'src/openjev/research/fsm_residual.py', 'src/openjev/research/fsm_shift_data.py',
         'research/fsm_author/src/openjev_fsm_author/__init__.py',
         'research/fsm_author/src/openjev_fsm_author/benchmark.py',
         'research/fsm_author/src/openjev_fsm_author/linear_context.py',
         'research/fsm_author/src/openjev_fsm_author/nllfr.py',
         'research/fsm_author/src/openjev_fsm_author/nllfr_context.py',
         'research/fsm_author/src/openjev_fsm_author/nllfr_context_budget.py',
         'scripts/audit_fsm_author_bla.py', 'scripts/audit_fsm_author_nllfr.py',
         'scripts/fsm_nllfr_budget_audit_math.py', 'pyproject.toml', 'uv.lock')
ENV = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
       'PYTHONPATH': 'src:research/fsm_author/src', 'PYTHONHASHSEED': '0',
       'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
TESTS = [*SOURCES[2:]]
COMMANDS = [
    ['.venv/bin/ruff', 'check', *SOURCES],
    ['.venv/bin/python', '-c', ('import sys, torch, pytest; torch.set_num_threads(1); '
     'torch.set_num_interop_threads(1); print("torch_threads", torch.get_num_threads(), '
     'torch.get_num_interop_threads()); sys.exit(pytest.main(sys.argv[1:]))'),
     '--noconftest', '-q', '--junitxml='+str(DEST/'junit.xml'), *TESTS],
]


def pin(path):
    payload = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.UTC).isoformat()
    before = {name: pin(ROOT/name) for name in (*SOURCES, *PRIOR)}
    for name, identity in before.items():
        snapshot = DEST/'source'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
        assert pin(snapshot) == identity
    (DEST/'qualify.py').write_bytes(Path(__file__).read_bytes())
    runtime = {'python': sys.version, 'executable': sys.executable,
               'platform': platform.platform(), 'machine': platform.machine(),
               'packages': {name: importlib.metadata.version(name)
                            for name in ('numpy', 'scipy', 'torch', 'pytest', 'ruff')}}
    scope = ('Fabricated array-only arithmetic and adapter qualification; no saved checkpoints, '
             'measurement archives, training or empirical shift evaluation.')
    save(DEST/'preflight.json', {'started_utc': started, 'sources': before,
         'commands': COMMANDS, 'thread_env': ENV, 'runtime': runtime,
         'per_command_timeout_seconds': 900, 'scope': scope})
    records, error = [], None
    try:
        for index, command in enumerate(COMMANDS, 1):
            log = DEST/f'command-{index:02d}.log'
            begin = time.perf_counter()
            record = {'command': command, 'log': str(log), 'returncode': None, 'timeout': False}
            try:
                with log.open('xb') as handle:
                    process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV},
                        stdout=handle, stderr=subprocess.STDOUT, timeout=900, check=False)
                    record['returncode'] = process.returncode
            except subprocess.TimeoutExpired:
                record['timeout'] = True
            finally:
                record.update(seconds=time.perf_counter()-begin, log_pin=pin(log))
                records.append(record)
            if record['returncode'] != 0:
                break
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        after = {name: pin(ROOT/name) for name in before}
        unchanged = before == after
        passed = error is None and unchanged and len(records) == len(COMMANDS) and all(
            row['returncode'] == 0 and not row['timeout'] for row in records)
        receipt = {'status': 'PASS' if passed else 'FAIL', 'started_utc': started,
                   'finished_utc': dt.datetime.now(dt.UTC).isoformat(), 'sources': before,
                   'sources_after': after, 'sources_unchanged': unchanged,
                   'prior_sources_unchanged': all(before[k] == after[k] for k in PRIOR),
                   'commands': records, 'runtime': runtime, 'thread_env': ENV,
                   'scope': scope, 'error': error}
        if (DEST/'junit.xml').exists():
            receipt['junit'] = pin(DEST/'junit.xml')
        save(DEST/'receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'commands': records,
                          'sources_unchanged': unchanged, 'receipt': str(DEST/'receipt.json')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
