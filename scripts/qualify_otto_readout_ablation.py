"""Bounded fabricated engineering checks and metadata handoff, no study fitting."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_readout_runner_qualification', ROOT / 'scripts/run_otto_readout_ablation.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.require(Path.cwd() == ROOT and args.output.is_absolute() and args.output.is_relative_to(ROOT), 'contained qualification output')
    args.output.mkdir(exist_ok=False)
    auth = runner.authenticate()
    before = auth['sources']
    tests = sorted(x for x in runner.NEW if x.startswith('tests/'))
    lint = sorted(x for x in runner.NEW if x.endswith('.py'))
    commands = [(180, [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                      '--basetemp', str(args.output / 'pytest-temp'), *tests]),
                (60, [str(ROOT / '.venv/bin/ruff'), 'check', '--no-cache', *lint]),
                (240, [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/qualify_otto_readout_capacity.py'),
                       '--output', str(args.output / 'capacity.json')])]
    env = {**os.environ, **{k: '1' for k in (*runner.THREADS, 'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD')}}
    runner.write(args.output / 'started.json', {'sources_before': before, 'commands': commands,
                                              'scope': 'fabricated data and metadata only', 'created_unix_ns': time.time_ns()})
    outcomes = []
    for i, (cap, command) in enumerate(commands):
        log = args.output / f'command-{i + 1}.log'
        started = time.monotonic()
        timed_out = False
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=5)
        try:
            os.killpg(child.pid, 0)
            absent = False
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            absent = True
        outcomes.append({'command': command, 'cap_seconds': cap, 'returncode': code, 'timed_out': timed_out,
                         'reaped': child.poll() is not None, 'group_absent': absent,
                         'elapsed_seconds': time.monotonic() - started, 'log': runner.descriptor(log)})
        if code != 0 or timed_out or not absent:
            break
    after = {name: runner.descriptor(name)['sha256'] for name in before}
    passed = len(outcomes) == 3 and all(x['returncode'] == 0 and not x['timed_out'] and x['group_absent'] for x in outcomes) and before == after
    # Exercise the actual stdlib-only runtime/lineage/bridge path after the checks.
    if passed:
        passed = runner.authenticate() == auth and not any(x in sys.modules for x in ('torch', 'numpy', 'tensorflow', 'jax', 'mlx'))
    files = {p.name: runner.descriptor(p) for p in args.output.iterdir() if p.is_file()}
    runner.write(args.output / 'receipt.json', {'status': 'passed' if passed else 'failed',
        'sources_before': before, 'sources_after': after, 'commands': outcomes, 'files': files,
        'metadata_handoff_passed': passed, 'empirical_array_decodes': 0, 'checkpoint_decodes': 0})
    print(json.dumps({'status': 'passed' if passed else 'failed', 'receipt': runner.descriptor(args.output / 'receipt.json')}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
