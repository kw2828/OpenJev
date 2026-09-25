"""Build a source-pinned robot inference library with an explicit Rust compiler.

No toolchain discovery, installation, dependency fetch, model load or benchmark.
The build directory is exclusive and retains failed compiler attempts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('rust/robot_transition/src/lib.rs',
           'rust/robot_transition/src/gru_reference.rs')


def descriptor(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Regular nonsymlink file required: ' + str(path))
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def build(rustc, output):
    compiler = Path(rustc).expanduser().absolute()
    compiler_pin = descriptor(compiler)
    if not os.access(compiler, os.X_OK):
        raise ValueError('Compiler must be executable')
    extension = {'darwin': 'dylib', 'linux': 'so', 'win32': 'dll'}.get(sys.platform)
    if extension is None:
        raise ValueError('Unsupported host platform')
    output = Path(output).resolve()
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    output.mkdir(parents=True, exist_ok=False)
    source_folder = output / 'source'
    source_folder.mkdir()
    for name, pin in sources.items():
        target = source_folder / Path(name).name
        with target.open('xb') as handle:
            handle.write((ROOT / name).read_bytes())
        if descriptor(target) != pin:
            raise ValueError('Source changed during snapshot')
    library = output / ('libopenjev_robot.' + extension)
    commands = [[str(compiler), '--version', '--verbose'],
                [str(compiler), '--edition=2021', '--crate-type=cdylib',
                 '--crate-name=openjev_robot', '-C', 'opt-level=3',
                 str(source_folder / 'lib.rs'), '-o', str(library)]]
    definition = {'created_utc': datetime.now(UTC).isoformat(),
                  'compiler': {'path': str(compiler), **compiler_pin},
                  'builder': descriptor(__file__), 'sources': sources,
                  'commands': commands,
                  'scope': 'Host library build only; no fast-math, dependencies, model execution or speed measurement'}
    write(output / 'definition.json', definition)
    processes = []
    for number, command in enumerate(commands):
        log = output / f'process-{number:02d}.log'
        started = time.monotonic()
        with log.open('xb') as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                     cwd=source_folder, check=False)
        processes.append({'command': command, 'returncode': process.returncode,
                          'seconds': time.monotonic() - started,
                          'log': {'path': str(log), **descriptor(log)}})
        if process.returncode:
            break
    passed = len(processes) == 2 and all(p['returncode'] == 0 for p in processes)
    unchanged = descriptor(compiler) == compiler_pin and all(descriptor(ROOT / p) == pin for p, pin in sources.items())
    record = {'status': 'PASS' if passed and unchanged else 'FAIL',
              'sources_unchanged': unchanged, 'processes': processes,
              'definition': descriptor(output / 'definition.json'),
              'library': {'path': str(library), **descriptor(library)} if library.is_file() else None}
    write(output / 'receipt.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rustc', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.rustc, args.output)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
