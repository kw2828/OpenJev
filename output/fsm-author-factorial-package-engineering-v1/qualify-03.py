"""One retained source/opaque-file qualification; no empirical inputs."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/fsm-author-factorial-package-engineering-v1/qualification-03'
SOURCES = ('scripts/package_fsm_author_factorial.py', 'output/fsm-author-factorial-package-engineering-v1/test_package.py')


def pin(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    before = {name: pin(ROOT/name) for name in SOURCES}
    snapshots = {}
    for name in SOURCES:
        target = OUT/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        snapshots[name] = pin(target)
    commands = [['.venv/bin/ruff', 'check', *SOURCES], ['.venv/bin/python', SOURCES[1]]]
    environment = {'PYTHONDONTWRITEBYTECODE': '1'}
    write(OUT/'preflight.json', {'sources': before, 'snapshots': snapshots, 'commands': commands,
        'environment': environment, 'wrapper': pin(Path(__file__)),
        'scope': 'Opaque temporary JSON/invalid NPZ files only. No empirical admission, models, scoring or arrays decoded.'})
    outcomes = []
    for i, command in enumerate(commands, 1):
        start = time.monotonic()
        log = OUT/f'command-{i:02d}.log'
        with log.open('xb') as stream:
            completed = subprocess.run(command, cwd=ROOT, env={**os.environ, **environment},
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        row = {'command': command, 'returncode': completed.returncode,
               'seconds': time.monotonic()-start, 'log': pin(log)}
        outcomes.append(row)
        write(OUT/f'command-{i:02d}.json', row)
        if completed.returncode:
            break
    after = {name: pin(ROOT/name) for name in SOURCES}
    passed = len(outcomes) == 2 and all(r['returncode'] == 0 for r in outcomes) and before == after
    receipt = {'status': 'PASS' if passed else 'FAIL', 'sources_before': before, 'sources_after': after,
               'sources_unchanged': before == after, 'preflight': pin(OUT/'preflight.json'), 'commands': outcomes}
    write(OUT/'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
