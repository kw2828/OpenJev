"""Prospective conditional-label training metadata and bounded-process helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-conditional-label-v1'
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
PARENT_VERSION = 'otto-cost-information-v1'
PARENT = ROOT / 'output' / PARENT_VERSION
PARENT_CLOSURE_PIN = '4e2aa1dc8dbe61f8c56722c0b547667c28b69bee504136b876ab13c3b8cf99b9'
FAMILIES = ('sampled', 'mean32')
FIT_SEEDS = (343000001, 343000002, 343000003)
CONFIG = {'prefix_transitions': 8, 'forecast_horizon': 8,
          'train_attempts': 512, 'dev_attempts_per_regime': 128,
          'train_draws': 32, 'dev_draws': 128, 'min_train': 256, 'min_dev_per_regime': 64,
          'train_first': 336000001, 'base_first': 337000001, 'shift_first': 338000001,
          'train_mc_first': 339000001, 'dev_mc_first': 340000001,
          'bootstrap_seed': 344000001, 'bootstrap_replicates': 2000,
          'quantile': .025, 'required_fraction': .05,
          'uniform_bits': 53, 'root_grid_max_tv': 1e-10,
          'epochs': 96, 'batch_size': 32, 'learning_rate': .003, 'clip_norm': 5.,
          'variance_floor': 1e-6, 'families': list(FAMILIES), 'fit_seeds': list(FIT_SEEDS)}
CAPS = {'collect': 3600, 'fit': 1800, 'audit': 3600}
SOURCES = ['scripts/qualify_otto_conditional_label.py', 'scripts/otto_conditional_label_common.py',
           'scripts/collect_otto_conditional_label.py', 'scripts/run_otto_conditional_label.py',
           'scripts/audit_otto_conditional_label.py',
           'src/openjev/research/otto_conditional_label_sampling.py',
           'src/openjev/research/otto_conditional_label.py',
           'tests/test_otto_conditional_label_sampling.py', 'tests/test_otto_conditional_label.py',
           'tests/test_run_otto_conditional_label.py', 'tests/test_audit_otto_conditional_label.py',
           'research/otto-conditional-label-protocol.md',
           'src/openjev/research/otto_action_latent_model.py',
           'src/openjev/research/otto_predictive_belief.py', 'src/openjev/research/otto_sampler_law.py',
           'tests/test_otto_action_latent_model.py', 'tests/test_otto_predictive_belief.py',
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
    collector = load(OLD_COLLECTOR, '_cost_information_native_lineage')
    from types import SimpleNamespace
    return collector.authenticate(SimpleNamespace(plan=ROOT / OLD_PLAN, plan_sha256=OLD_PLAN_PIN))


def roster():
    rows = []
    for split, regime, first, count in (
        ('train', 'lambda3', CONFIG['train_first'], CONFIG['train_attempts']),
        ('dev', 'lambda3', CONFIG['base_first'], CONFIG['dev_attempts_per_regime']),
        ('dev', 'lambda4', CONFIG['shift_first'], CONFIG['dev_attempts_per_regime']),
    ):
        for i in range(count):
            ordinal = i if split == 'train' else len(rows) - CONFIG['train_attempts']
            mc_first = CONFIG['train_mc_first'] if split == 'train' else CONFIG['dev_mc_first']
            rows.append({'split': split, 'regime': regime, 'seed': first + i,
                         'initial_hit': 1 + i % 3, 'mc_seed': mc_first + ordinal,
                         'id': f'{split}:{regime}:{first + i}'})
    return rows


def parent_reference():
    """Metadata-only evidence for the preceding headroom criterion, no arrays."""
    path = PARENT / 'closure-01.json'
    require(desc(path)['sha256'] == PARENT_CLOSURE_PIN, 'published headroom closure')
    closure = read(path)
    require(closure['version'] == PARENT_VERSION and closure['technical_complete']
            and closure['independent_audit_passed'] and not closure['old_test_admitted'],
            'audited headroom diagnostic')
    summary = ROOT / 'research/otto-cost-information-results/summary.json'
    require(desc(summary)['sha256'] == '4c83e72211d3f6f411d4becc7d0702a11cd83fec3c4ee1fb580c84ffe3097932',
            'published scalar report')
    require(closure['status'] == 'HEADROOM_RESOLVED', 'headroom resolved')
    return {'closure': desc(path), 'summary': desc(summary),
            'scientific_status': closure['status'],
            'parent_arrays_decoded': 0, 'parent_checkpoints_decoded': 0,
            'scope': 'Hypothesis evidence only. New models and labels use fresh data.'}


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
        require(all(os.environ.get(k) == '1' for k in THREADS), 'single numerical threads')
        require(Path.cwd() == ROOT and self.out.is_absolute() and self.out.is_relative_to(OUT), 'study output')
        stem = {'collect': 'collection', 'fit': 'fit', 'audit': 'audit'}[phase]
        require(self.out == OUT / (stem + '-01')
                and args.supervision == OUT / (stem + '-native-01.launch.json')
                and args.plan == OUT / 'registration-01.json',
                'one original registered attempt per phase')
        self.out.mkdir(exist_ok=False)
        self.clock = load('src/openjev/research/suspend_clock.py', '_cost_information_clock').SuspendClock()
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
        if phase in ('fit', 'audit'):
            runtime = self.plan['numerical_runtime']
            require(sys.executable == runtime['executable'] and sys.version == runtime['python']
                    and all(importlib.metadata.version(name) == version
                            for name, version in runtime['packages'].items()),
                    'unchanged registered numerical runtime')
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
