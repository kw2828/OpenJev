"""Prospective action-gap pilot metadata and bounded-process helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-action-latent-v1'
OUT = ROOT / 'output' / VERSION
NATIVE = ROOT / '.venv-otto-released-native/bin/python'
NUMERICAL = ROOT / '.venv/bin/python'
OLD_PLAN = 'output/otto-query-gate-learning-v1/collection-plan-01.json'
OLD_PLAN_PIN = 'f16f23f82d9eb271ac3c1a15df0c567f0a782b169abcdd4711de8fb9a7700be6'
OLD_COLLECTOR = 'scripts/collect_otto_query_gate.py'
OLD_COLLECTOR_PIN = '5e2b06fc0ff5b888122e733f12b20c881deab7f56b94908ed5a283f7a48c76df'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
CONFIG = {'prefix_transitions': 8, 'train_horizon': 4, 'dev_horizon': 8,
          'train_cases': 192, 'dev_cases_per_regime': 48, 'min_train': 120, 'min_dev_per_regime': 24, 'minimum_supported_cases': 16,
          'train_first': 320000001, 'dev_base_first': 321000001, 'dev_shift_first': 322000001,
          'fit_seeds': [323000001, 323000002, 323000003], 'epochs': 80, 'batch': 32,
          'lr': .003, 'clip': 5., 'aux_weight': .1, 'cost_weight': 1.,
          'ridge_lambda': .0001, 'ridge_probability_smoothing': .000001,
          'candidate': 'action_recurrent', 'controls': ['action_blind', 'direct_horizon', 'ridge'],
          'long_logscore_improvement': .01, 'long_gap_improvement': .05,
          'normal_relative_tolerance': .01}
CAPS = {'collect': 1800, 'fit': 1800, 'audit': 180}
SOURCES = ['scripts/qualify_otto_action_latent.py', 'scripts/otto_action_latent_common.py', 'scripts/collect_otto_action_latent.py',
           'scripts/run_otto_action_latent.py', 'scripts/audit_otto_action_latent.py',
           'src/openjev/research/otto_action_latent_model.py',
           'src/openjev/research/otto_action_latent_metrics.py',
           'src/openjev/research/otto_action_latent_ridge.py',
           'tests/test_otto_action_latent_model.py', 'tests/test_otto_action_latent_metrics.py',
           'tests/test_otto_action_latent_ridge.py', 'tests/test_collect_otto_action_latent.py',
           'tests/test_run_otto_action_latent.py', 'tests/test_audit_otto_action_latent.py',
           'research/otto-action-latent-protocol.md',
           'scripts/supervise_dialogue_observation_v2.py', 'src/openjev/research/suspend_clock.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and not path.is_symlink(), 'regular input')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(value):
    path = Path(value)
    return json.loads((path if path.is_absolute() else ROOT / path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(value, name):
    path = Path(value)
    spec = importlib.util.spec_from_file_location(name, path if path.is_absolute() else ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_native():
    require(desc(OLD_COLLECTOR)['sha256'] == OLD_COLLECTOR_PIN, 'qualified collector source')
    require(desc(OLD_PLAN)['sha256'] == OLD_PLAN_PIN, 'qualified collector plan')
    collector = load(OLD_COLLECTOR, '_action_latent_native_lineage')
    from types import SimpleNamespace
    return collector.authenticate(SimpleNamespace(plan=ROOT / OLD_PLAN, plan_sha256=OLD_PLAN_PIN))


def roster():
    result = []
    for split, regime, first, count in (
        ('train', 'lambda3', CONFIG['train_first'], CONFIG['train_cases']),
        ('dev', 'lambda3', CONFIG['dev_base_first'], CONFIG['dev_cases_per_regime']),
        ('dev', 'lambda4', CONFIG['dev_shift_first'], CONFIG['dev_cases_per_regime'])):
        for i in range(count):
            result.append({'split': split, 'regime': regime, 'seed': first + i,
                           'initial_hit': 1 + i % 3, 'id': f'{split}:{regime}:{first + i}'})
    return result


def check_plan(plan):
    require(plan['version'] == VERSION and plan['config'] == CONFIG and plan['roster'] == roster(), 'fixed protocol')
    for name, expected in plan['sources'].items():
        require(desc(name) == expected, 'unchanged source: ' + name)
    require(set(SOURCES) <= set(plan['sources']), 'complete new source closure')
    for name, expected in plan['inputs'].items():
        require(desc(name) == expected, 'unchanged input: ' + name)


class Run:
    def __init__(self, args, phase):
        self.args, self.phase, self.out = args, phase, args.output
        require(Path.cwd() == ROOT and self.out.is_absolute() and self.out.is_relative_to(OUT), 'study output')
        self.out.mkdir(exist_ok=False)
        self.clock = load('src/openjev/research/suspend_clock.py', '_action_latent_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5e9, 'launch receipt wait')
            time.sleep(.01)
        self.launch = read(args.supervision)
        require(self.launch['command'] == [sys.executable, *sys.argv]
                and self.launch['pid'] == os.getpid() == self.launch['pgid'] == os.getpgrp()
                and self.launch['parent_pid'] == os.getppid()
                and self.launch['cap_seconds'] == CAPS[phase]
                and self.launch['clock_backend'] == self.clock.backend
                and self.launch['cwd'] == str(Path.cwd()) == str(ROOT)
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + CAPS[phase] * 10**9,
                'original bounded process')
        require(desc(args.plan)['sha256'] == args.plan_sha256, 'external registration pin')
        self.plan = read(args.plan)
        check_plan(self.plan)
        require(self.launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
                and self.launch['clock_source_sha256'] == self.plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
                'registered supervisor sources')
        self.receipt = {'version': VERSION, 'phase': phase, 'status': 'started',
                        'plan_sha256': args.plan_sha256, 'supervision': desc(args.supervision),
                        'started_ns': self.start, 'calls': {}, 'old_test_decodes': 0, 'astra_calls': 0}
        write(self.out / 'started.json', {'launch': self.launch, 'plan_sha256': args.plan_sha256})
        self.check()

    def check(self):
        require(self.clock.now_ns() < self.launch['deadline_ns'], 'fixed suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        require(rss <= 4 * 1024**3, '4GiB RSS cap')
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= 512 * 1024**2, '512MiB output cap')
        self.peak_rss = rss

    def finish(self, error=None):
        if error is None:
            check_plan(self.plan)
            self.check()
        self.receipt.update(status='completed' if error is None else 'failed', error=None if error is None else repr(error),
                            finished_ns=self.clock.now_ns(), peak_rss_bytes=self.peak_rss,
                            files={p.name: desc(p) for p in self.out.iterdir() if p.is_file()})
        self.receipt['wall_seconds'] = (self.receipt['finished_ns'] - self.start) / 1e9
        write(self.out / 'receipt.json', self.receipt)


def closed(path, terminal_path):
    receipt = read(path / 'receipt.json')
    start = read(path / 'started.json')
    terminal = read(terminal_path)
    require(receipt['status'] == 'completed' and terminal['status'] == 'completed'
            and terminal['returncode'] == 0 and terminal['group_absent'] and terminal['cleanup']['reaped']
            and not terminal['timed_out'] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True
            and terminal['cleanup']['group_absent'] and terminal['cleanup']['errors'] == []
            and terminal['finished_ns'] <= terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + terminal['cap_seconds'] * 10**9
            and all(terminal[k] == v for k, v in start['launch'].items())
            and start['launch']['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns'],
            'successful original process closure')
    require(receipt['version'] == VERSION and receipt['phase'] in CAPS
            and terminal['cap_seconds'] == CAPS[receipt['phase']]
            and receipt['plan_sha256'] == start['plan_sha256'], 'closed phase and registration')
    launch_index = terminal['command'].index('--supervision') + 1
    require(desc(terminal['command'][launch_index]) == receipt['supervision'], 'original launch descriptor')
    require({p.name for p in path.iterdir()} == set(receipt['files']) | {'receipt.json'}, 'closed inventory')
    for name, expected in receipt['files'].items():
        require(desc(path / name) == expected, 'closed artifact: ' + name)
    return receipt
