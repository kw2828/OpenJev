"""Bounded fabricated qualification of a prospective operator component only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/otto-observation-operator-component-v1'
SOURCES = ['scripts/qualify_otto_observation_operator.py',
           'src/openjev/research/otto_observation_operator_model.py',
           'tests/test_otto_observation_operator_model.py',
           'research/otto-observation-operator-component.md',
           'research/otto-predictive-moment-mechanism-draft.md']
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def preserve_running_study():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import otto_conditional_label_common as c
    registration = c.OUT / 'registration-01.json'
    require(c.desc(registration)['sha256'] == 'fc03fafdf7fc3cff7b25350d6baeb02323ca0f4da31425fcf42f86db215dc5f4',
            'unchanged separately running experiment registration')
    c.check_plan(c.read(registration))
    return c.desc(registration)


def geometry(output):
    require(output.is_absolute() and output == output.resolve() and output.is_relative_to(OUT.resolve()),
            'canonical contained component geometry output')
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from openjev.research.otto_observation_operator_model import make_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(714)
        for kind in ('tied', 'untied'):
            model = make_model(kind, seed=715, width=14, cost_scale=1.)
            by_dtype = {}
            for parameter in model.parameters():
                name = str(parameter.dtype)
                item = by_dtype.setdefault(name, {'parameters': 0, 'bytes': 0})
                item['parameters'] += parameter.numel()
                item['bytes'] += parameter.numel() * parameter.element_size()
            for batch in (1, 32):
                prefix = torch.randn(batch, 9, 31, dtype=torch.float32)
                lengths = torch.full((batch,), 9, dtype=torch.int64)
                for horizon in (1, 8):
                    actions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3][:horizon]], dtype=torch.int64).expand(batch, -1).clone()
                    odors = torch.tensor([[0, 1, 2, 3, 4, 4, 4, 4][:horizon]], dtype=torch.int64).expand(batch, -1).clone()
                    with torch.no_grad():
                        blind = model.blind_rollout(prefix, lengths, actions)
                        observed = model.observed_rollout(prefix, lengths, actions, odors)
                    for result in (blind, observed):
                        require(tuple(result['cost_contrasts'].shape) == (batch, horizon, 4)
                                and bool(torch.isfinite(result['cost_contrasts']).all()), 'finite complete fabricated forecasts')
                    records.append({'kind': kind, 'batch': batch, 'prefix_length': 9, 'horizon': horizon,
                                    'parameter_dtypes': by_dtype, 'carried_state_bytes_per_row': 14 * 8})
    write(output, {'status': 'passed', 'geometry': records, 'rollouts': 16,
        'empirical_data_decodes': 0, 'simulator_calls': 0, 'teacher_calls': 0,
        'empirical_training_updates': 0,
        'scope': 'Random untrained models and fabricated observations only. No runtime or architecture advantage measured.'})


def qualify(output):
    require(Path.cwd() == ROOT and output.is_absolute() and output == output.resolve()
            and output.is_relative_to(OUT.resolve()), 'canonical contained component output')
    registration = preserve_running_study()
    before = {name: desc(ROOT / name) for name in SOURCES}
    output.mkdir(parents=True, exist_ok=False)
    commands = [(180, [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                      '--basetemp', str(output / 'pytest-temp'), 'tests/test_otto_observation_operator_model.py']),
                (60, [str(ROOT / '.venv/bin/ruff'), 'check', '--no-cache', *[p for p in SOURCES if p.endswith('.py')]]),
                (90, [str(ROOT / '.venv/bin/python'), str(ROOT / SOURCES[0]), '--geometry', '--output', str(output / 'geometry.json')])]
    write(output / 'started.json', {'sources': before, 'commands': commands, 'running_study_registration': registration,
        'scope': 'Fabricated structural qualification only; no new empirical trial admitted.'})
    env = {**os.environ, **dict.fromkeys((*THREADS, 'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD'), '1')}
    outcomes = []
    for index, (cap, command) in enumerate(commands):
        log = output / f'command-{index + 1}.log'
        tick, timed_out = time.monotonic(), False
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
        try:
            os.killpg(process.pid, 0)
            absent = False
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            absent = True
        outcomes.append({'command': command, 'cap_seconds': cap, 'returncode': code, 'timed_out': timed_out,
                         'group_absent': absent, 'reaped': process.poll() is not None,
                         'seconds': time.monotonic() - tick, 'log': log.name, **desc(log)})
        if code or timed_out or not absent:
            break
    after = {name: desc(ROOT / name) for name in SOURCES}
    preservation_error = None
    try:
        preserved = preserve_running_study() == registration
    except (ValueError, OSError, KeyError, TypeError) as exc:
        preserved, preservation_error = False, repr(exc)
    passed = before == after and preserved and len(outcomes) == 3 and all(
        r['returncode'] == 0 and not r['timed_out'] and r['group_absent'] and r['reaped'] for r in outcomes)
    write(output / 'receipt.json', {'status': 'passed' if passed else 'failed', 'source_before': before,
        'source_after': after, 'scientific_registration_unchanged': preserved,
        'preservation_error': preservation_error,
        'commands': outcomes, 'files': {p.name: desc(p) for p in output.iterdir() if p.is_file()},
        'scope': 'Tests and geometry are fabricated. No empirical effectiveness, speed, calibration or novelty claim.',
        'admits_empirical_execution': False})
    print(json.dumps({'status': 'passed' if passed else 'failed', 'output': str(output)}))
    require(passed, 'complete fabricated component qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', action='store_true')
    arguments = parser.parse_args()
    if arguments.geometry:
        geometry(arguments.output)
    else:
        qualify(arguments.output)
