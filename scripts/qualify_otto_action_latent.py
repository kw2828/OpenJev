"""Fabricated engineering checks and prospective one-epoch capacity bound."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import otto_action_latent_common as c


def capacity(path):
    import numpy as np
    import torch
    sys.path.insert(0, str(c.ROOT / 'src'))
    from run_otto_action_latent import loss_for

    from openjev.research.otto_action_latent_model import make_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(202)
    n, h = 192, 4
    data = {'prefix': torch.from_numpy(rng.normal(size=(n, 9, 31)).astype(np.float32)),
            'prefix_lengths': torch.full((n,), 9, dtype=torch.int64),
            'actions': torch.from_numpy(rng.integers(0, 4, (n, h), dtype=np.int64)),
            'continuation': torch.from_numpy(rng.normal(size=(n, h, 31)).astype(np.float32)),
            'outcomes': torch.from_numpy(rng.integers(0, 4, (n, h), dtype=np.int64)),
            'raw_costs': torch.from_numpy(rng.normal(size=(n, h, 4)).astype(np.float32))}
    times = {}
    for family in ('action_recurrent', 'action_blind', 'direct_horizon'):
        tick = time.perf_counter()
        model = make_model(family, 17, cost_scale=.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        for start in range(0, n, 32):
            batch = {k: v[start:start + 32] for k, v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            gap = model.blind_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'])
            normal = model.normal_rollout(batch['prefix'], batch['prefix_lengths'], batch['actions'], batch['continuation'], found=batch['outcomes'] == 4)
            loss = (loss_for(gap, batch, torch, .01) + loss_for(normal, batch, torch, .01)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        times[family] = time.perf_counter() - tick
    projected = 2 * 3 * 80 * sum(times.values()) + 180
    result = {'status': 'passed' if projected <= 1800 else 'failed', 'one_epoch_seconds': times,
              'pessimistic_projection_seconds': projected, 'cap_seconds': 1800,
              'fabricated_cases': n, 'epochs_per_arm': 1, 'empirical_decodes': 0}
    c.write(path, result)
    c.require(result['status'] == 'passed', 'prospective capacity gate')


def qualify(output):
    output.mkdir(exist_ok=False)
    before = {k: c.desc(k) for k in c.SOURCES}
    tests = [k for k in c.SOURCES if k.startswith('tests/')]
    lint = [k for k in c.SOURCES if k.endswith('.py') and not k.endswith(('suspend_clock.py', 'supervise_dialogue_observation_v2.py'))]
    commands = [(180, [str(c.NUMERICAL), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(output / 'pytest-temp'), *tests]),
                (60, [str(c.ROOT / '.venv/bin/ruff'), 'check', '--no-cache', *lint]),
                (120, [str(c.NUMERICAL), str(Path(__file__).absolute()), '--capacity', '--output', str(output / 'capacity.json')])]
    c.write(output / 'started.json', {'sources': before, 'commands': commands, 'scope': 'fabricated qualification only'})
    outcomes = []
    env = {**os.environ, **dict.fromkeys((*c.THREADS, 'PYTHONDONTWRITEBYTECODE', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD'), '1')}
    for i, (cap, command) in enumerate(commands):
        tick = time.perf_counter()
        timeout = False
        log = output / f'command-{i + 1}.log'
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
        try:
            os.killpg(process.pid, 0)
            absent = False
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            absent = True
        outcomes.append({'command': command, 'returncode': code, 'timed_out': timeout, 'reaped': process.poll() is not None,
                         'group_absent': absent, 'seconds': time.perf_counter() - tick, 'log': log.name, **c.desc(log)})
        if code or timeout or not absent:
            break
    after = {k: c.desc(k) for k in c.SOURCES}
    passed = before == after and len(outcomes) == len(commands) and all(x['returncode'] == 0 and not x['timed_out'] and x['group_absent'] for x in outcomes)
    c.write(output / 'receipt.json', {'status': 'passed' if passed else 'failed', 'source_before': before, 'source_after': after,
            'commands': outcomes, 'files': {p.name: c.desc(p) for p in output.iterdir() if p.is_file()}, 'empirical_decodes': 0})
    print({'status': 'passed' if passed else 'failed', 'output': str(output)}, flush=True)
    c.require(passed, 'complete source qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity', action='store_true')
    args = parser.parse_args()
    capacity(args.output) if args.capacity else qualify(args.output)
