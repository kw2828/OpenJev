"""Registered direct readout solve; canonical TRAIN surrogate, reused DEV diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/run_otto_direct_readout.py'
VERSION = 'otto-direct-readout-v1'
BASE = 'output/otto-direct-readout-v1'
PRIOR = 'scripts/run_otto_readout_compute.py'
PRIOR_SHA = '47048d43fb76802213c33d42916745f6635fc8ff87bbd0b9a19d8afe48593ebb'
CLOSURE = 'output/otto-readout-compute-v1/closure-01.json'
CLOSURE_SHA = '8ba451fb8f8e46c60e08d453f2b7be5808b3010f6af65bf33d8563e2b30da518'
CLOCK = 'src/openjev/research/suspend_clock.py'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
SEEDS = (309000001, 309000002, 309000003)
ARMS = ('ols', 'ridge')
TRAIN_VIEWS = ('pretrained', 'action_residual_only', *ARMS)
VIEWS = (*TRAIN_VIEWS, 'full_joint')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
LIMITS = {'train': {'seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 1024**3},
          'audit': {'seconds': 600, 'rss_bytes': 2 * 1024**3, 'output_bytes': 512 * 1024**2}}
CONFIG = {'seeds': list(SEEDS), 'solvers': list(ARMS), 'ridge': .0001, 'rcond': 1e-10,
          'batch': 1, 'chunk': 32, 'query_period': 4, 'solves': 6, 'train_views': 12, 'dev_views': 15,
          'train_episodes': 54, 'dev_episodes': 36, 'parity_atol': 1e-5, 'parity_rtol': 1e-5,
          'candidate': 'ridge', 'controls': ['pretrained', 'action_residual_only'], 'margin': .05,
          'dev_reused': True, 'fresh_dev': False, 'held_out_from_training': True,
          'test_admitted': False, 'confirmation_admitted': False, 'held_out_evidence': False}
NEW = {SELF, 'scripts/audit_otto_direct_readout.py', 'scripts/qualify_otto_direct_readout.py',
       'src/openjev/research/otto_direct_readout.py', 'src/openjev/research/otto_direct_readout_cache.py',
       'tests/test_otto_direct_readout.py', 'tests/test_otto_direct_readout_cache.py',
       'tests/test_run_otto_direct_readout.py', 'tests/test_audit_otto_direct_readout.py',
       'research/otto-direct-readout-protocol.md'}

def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and '..' not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), 'regular contained input')
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(record):
    require(descriptor(record['path']) == record, 'unchanged exact input descriptor')



def authenticate():
    require(descriptor(PRIOR)['sha256'] == PRIOR_SHA and descriptor(CLOSURE)['sha256'] == CLOSURE_SHA,
            'immutable completed predecessor')
    old = load(PRIOR, '_direct_prior')
    closure = read(CLOSURE)
    plan, receipt, _ = old.process_closure(closure['plan']['path'], closure['producer_receipt']['path'],
                                          closure['producer_terminal']['path'])
    old.process_closure(closure['plan']['path'], closure['audit_receipt']['path'], closure['audit_terminal']['path'])
    require(closure['technical_complete'] and closure['status'] == 'DEV_FAIL', 'closed prior failure retained')
    sources = {**plan['sources'], **{name: descriptor(name)['sha256'] for name in NEW}}
    comparators = {family: {str(seed): receipt['files'][f'checkpoint-{family}-{seed}.npz'] for seed in SEEDS}
                   for family in ('action_residual_only', 'full_joint')}
    for items in comparators.values():
        for pin in items.values():
            verify(pin)
    return {**{k: plan[k] for k in ('train', 'dev', 'lineage', 'runtime')}, 'sources': sources,
            'comparators': comparators, 'prior_closure': descriptor(CLOSURE),
            'prior_fits': receipt['files']['fits.json'], 'prior_views': receipt['files']['views.json']}


def engineering(path):
    record = read(path)
    require(record['status'] == 'passed' and record['sources_before'] == record['sources_after']
            and len(record['commands']) == 3 and all(c['returncode'] == 0 and not c['timed_out']
                and c['reaped'] and c['group_absent'] for c in record['commands'])
            and record['empirical_array_decodes'] == record['checkpoint_decodes'] == record['scientific_calls'] == 0,
            'completed fabricated engineering qualification')
    require(NEW <= set(record['sources_after']), 'all new sources qualified')
    for name, sha in record['sources_after'].items():
        require(descriptor(name)['sha256'] == sha, 'qualified source unchanged')
    for pin in record['files'].values():
        verify(pin)
    capacity = read(record['files']['capacity.json']['path'])
    require(capacity['status'] == 'passed' and capacity['projected_seconds'] <= 1350
            and capacity['audit_projected_seconds'] <= 450
            and capacity['empirical_array_decodes'] == capacity['checkpoint_decodes'] == 0,
            'bounded fabricated capacity')
    return descriptor(path)


def validate_plan(plan):
    require(plan['version'] == VERSION and plan['configuration'] == CONFIG and plan['limits'] == LIMITS
            and plan['status'] == 'registered_before_empirical_decodes', 'fixed new diagnostic recipe')
    actual = authenticate()
    require(all(plan[k] == v for k, v in actual.items()), 'unchanged registered evidence and sources')
    require(engineering(plan['engineering']['path']) == plan['engineering'], 'bound qualification')


def freeze(args):
    auth = authenticate()
    qualification = engineering(args.engineering)
    require(not any(n in sys.modules for n in ('numpy', 'torch', 'tensorflow')), 'metadata-only registration')
    write(args.output, {'version': VERSION, 'status': 'registered_before_empirical_decodes',
                       'created_unix_ns': time.time_ns(), 'configuration': CONFIG, 'limits': LIMITS,
                       'engineering': qualification, **auth})
    print(json.dumps(descriptor(args.output)), flush=True)


def expected_payloads(phase):
    if phase == 'audit':
        return {'started.json', 'runtime.json', 'audit.json'}
    require(phase == 'train', 'declared phase')
    return {'started.json', 'runtime.json', 'progress.jsonl', 'fits.json', 'views.json',
            'training-barrier.json', 'summary.json', 'train-history.npz', 'train-history.json',
            'dev-history.npz', 'dev-history.json'} | {f'cache-{s}.npz' for s in SEEDS} | {
            f'checkpoint-{a}-{s}.npz' for s in SEEDS for a in ARMS} | {
            f'train-prediction-{a}-{s}.npz' for s in SEEDS for a in TRAIN_VIEWS} | {
            f'dev-prediction-{a}-{s}.npz' for s in SEEDS for a in VIEWS}


def process_closure(planpath, receiptpath, terminalpath):
    plan, receipt, terminal = read(planpath), read(receiptpath), read(terminalpath)
    validate_plan(plan)
    phase = receipt['phase']
    require(phase in LIMITS and receipt['version'] == VERSION and receipt['status'] == 'completed'
            and receipt['pending'] is None and receipt['plan'] == descriptor(planpath), 'complete original worker')
    launch = read(receipt['supervision']['path'])
    verify(receipt['supervision'])
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and not terminal['timed_out'] and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['cleanup']['reaped'] and terminal['cleanup']['group_absent']
            and not terminal['cleanup']['errors'] and terminal['group_absent'] and terminal['timing_available']
            and terminal['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and all(terminal[k] == v for k, v in launch.items()), 'successful original supervisor closure')
    require(launch['pid'] == launch['pgid'] != launch['parent_pid'] and Path(launch['cwd']) == ROOT
            and launch['cap_seconds'] == LIMITS[phase]['seconds']
            and launch['deadline_ns'] == launch['started_ns'] + LIMITS[phase]['seconds'] * 10**9
            and launch['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns']
            < launch['deadline_ns'] and launch['watchdog_sha256'] == plan['sources'][SUPERVISOR]
            and launch['clock_source_sha256'] == plan['sources'][CLOCK], 'same original bounded process')
    directory = regular(receiptpath).parent
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / SELF), phase, '--plan', str(regular(planpath)),
               '--plan-sha256', descriptor(planpath)['sha256'], '--supervision', receipt['supervision']['path'],
               '--output', str(directory), *receipt.get('extra_arguments', [])]
    require(launch['command'] == command and receipt['sources'] == plan['sources']
            and receipt['limits'] == LIMITS[phase]
            and receipt['teacher_calls'] == receipt['native_calls'] == receipt['test_array_decodes'] == 0
            and receipt['optimizer_steps'] == 0, 'exact admitted command, sources and no extra scientific calls')
    require(set(receipt['files']) == expected_payloads(phase)
            and {p.name for p in directory.iterdir()} == expected_payloads(phase) | {'receipt.json'}, 'exact payload roster')
    for name, pin in receipt['files'].items():
        require(Path(pin['path']) == directory / name, 'contained payload')
        verify(pin)
    started = read(directory / 'started.json')
    require(started['launch'] == launch and started['started_ns'] == receipt['started_ns'], 'bound start')
    if phase == 'train':
        require(receipt['solves_completed'] == receipt['fits_completed'] == 6
                and receipt['train_views_completed'] == 12 and receipt['views_completed'] == 15
                and receipt['checkpoint_decodes'] == 9 and receipt['array_decodes'] == 11, 'all new fits and views')
    else:
        counts = receipt['audit_counts']
        require(receipt['agreement'] and receipt['array_decodes'] == counts['array_decodes'] == 49
                and receipt['checkpoint_decodes'] == counts['checkpoint_decodes'] == 15
                and receipt['views_completed'] == counts['views_completed'] == 27
                and counts['fits_checked'] == 6 and receipt['train_views_completed'] == counts['train_views_completed'] == 12
                and counts['dev_views_completed'] == 15, 'complete independent audit counters')
        verify(receipt['producer_receipt'])
        verify(receipt['producer_terminal'])
        require(receipt['extra_arguments'] == ['--producer-receipt', receipt['producer_receipt']['path'],
                '--producer-terminal', receipt['producer_terminal']['path']], 'bound audit producer arguments')
    return plan, receipt, directory
class BoundRun:
    def __init__(self, args):
        self.args, self.phase, self.out = args, args.mode, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.owns_output = False
        self.receipt = {'version': VERSION, 'phase': self.phase, 'status': 'started', 'pending': None,
                        'fits_completed': 0, 'optimizer_steps': 0, 'episode_exposures': 0,
                        'teacher_calls': 0, 'native_calls': 0, 'test_array_decodes': 0,
                        'checkpoint_decodes': 0, 'array_decodes': 0, 'views_completed': 0,
                        'train_views_completed': 0, 'solves_completed': 0, 'counter_scope': 'completed returned operations; pending stage preserves uncertain partial work'}

    def initialize(self):
        self.out.mkdir(exist_ok=False)
        self.owns_output = True
        self.clock = load(CLOCK, '_readout_clock').SuspendClock()
        self.start = self.clock.now_ns()
        self.receipt['started_ns'] = self.start

    def check(self):
        if self.launch is not None:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'fixed original worker deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS[self.phase]['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir()) < LIMITS[self.phase]['output_bytes'] - 1024**2, 'output cap with failure reserve')

    def bind(self):
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'original launch available')
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        require(self.launch['command'] == [sys.executable, *sys.argv]
                and self.launch['pid'] == os.getpid() and self.launch['pgid'] == os.getpgrp()
                and self.launch['parent_pid'] == os.getppid() and Path.cwd() == ROOT
                and self.launch['cap_seconds'] == LIMITS[self.phase]['seconds']
                and self.launch['clock_backend'] == self.clock.backend
                and self.launch['started_ns'] <= self.start < self.launch['deadline_ns']
                and self.launch['deadline_ns'] == self.launch['started_ns'] + LIMITS[self.phase]['seconds'] * 10**9,
                'original detached worker identity')
        require(descriptor(self.args.plan)['sha256'] == self.args.plan_sha256, 'external plan pin')
        self.plan = read(self.args.plan)
        validate_plan(self.plan)
        require(self.launch['clock_source_sha256'] == self.plan['sources'][CLOCK]
                and self.launch['watchdog_sha256'] == self.plan['sources'][SUPERVISOR], 'bound supervisor code')
        require(all(os.environ.get(k) == '1' for k in THREADS), 'one numerical CPU thread')
        self.receipt.update(plan=descriptor(self.args.plan), supervision=descriptor(self.args.supervision),
                            sources=self.plan['sources'], limits=LIMITS[self.phase])
        self.publish('started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                     'launch': self.launch, 'started_ns': self.start})
        self.publish('runtime.json', self.plan['runtime'])
        self.check()

    def publish(self, name, value):
        self.check()
        write(self.out / name, value)
        return descriptor(self.out / name)

    def event(self, value, filename='work.jsonl'):
        require(filename in ('work.jsonl', 'progress.jsonl'), 'declared journal')
        self.check()
        with (self.out / filename).open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def save_arrays(self, name, arrays):
        self.check()
        with (self.out / name).open('xb') as stream:
            self.np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return descriptor(self.out / name)

    def decode(self, pin):
        self.check()
        verify(pin)
        self.receipt['pending'] = {'decode': pin}
        with self.np.load(pin['path'], allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        self.receipt['array_decodes'] += 1
        self.receipt['pending'] = None
        return result


    def history(self, stage):
        require(stage in ('train', 'dev'), 'TRAIN and reused DEV only')
        if stage == 'dev':
            require(self.receipt['solves_completed'] == 6 and self.receipt['train_views_completed'] == 12
                    and (self.out / 'training-barrier.json').is_file(), 'all solves and parity checks before DEV')
        from openjev.research import otto_query_memory_data as data
        record = self.plan[stage]
        history = data.project_census(self.decode(record['descriptor']), record['identities'],
                                      query_period=4, expected_stage=stage)
        self.save_arrays(stage + '-history.npz', {k: v for k, v in history.items() if isinstance(v, self.np.ndarray)})
        self.publish(stage + '-history.json', {k: v for k, v in history.items() if not isinstance(v, self.np.ndarray)})
        return history

    def state(self, pin):
        raw = self.decode(pin)
        self.receipt['checkpoint_decodes'] += 1
        require(all(k.startswith('slow.') for k in raw), 'slow-only checkpoint')
        return {k.removeprefix('slow.'): self.torch.from_numpy(v.copy()) for k, v in raw.items()}

    def theta(self, state):
        return self.np.column_stack((state['action_residual.weight'].numpy(),
                                     state['action_residual.bias'].numpy())).astype(self.np.float64)

    def train(self):
        import numpy as np
        import torch

        from openjev.research import otto_direct_readout as solver
        from openjev.research import otto_direct_readout_cache as cache_module
        self.np, self.torch = np, torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        history = self.history('train')
        states, fits, views = {}, [], []
        for seed in SEEDS:
            parent = self.state(self.plan['lineage']['checkpoints']['pretrained'][str(seed)])
            states['pretrained', seed] = parent
            for family in ('action_residual_only', 'full_joint'):
                states[family, seed] = self.state(self.plan['comparators'][family][str(seed)])
            frozen = [k for k in parent if not k.startswith('action_residual.')]
            require(len(frozen) == 6 and all(torch.equal(parent[k], states['action_residual_only', seed][k])
                    for k in frozen), 'Adam comparator shares six frozen parent tensors')
            self.receipt['pending'] = {'seed': seed, 'stage': 'TRAIN feature extraction'}
            tick = self.clock.now_ns()
            cache = cache_module.extract(parent, seed, history, check=self.check)
            helper_seconds = cache['seconds']
            cache['seconds'] = (self.clock.now_ns() - tick) / 1e9
            theta = self.theta(parent)
            tick = self.clock.now_ns()
            legal = history['legal'].copy()
            legal[history['prior_mask']] = True
            weights = history['nonquery_weights'] / legal.sum(1) + history['prior_weights'] / 4
            # The explicit float64 surrogate uses the raw base and exact promoted
            # head, rather than subtracting rounded float32 model predictions.
            parent_affine = cache_module.affine_prediction(cache, theta)
            target_error = history['targets'].astype(np.float64) / 64 - parent_affine
            A, b = solver.build_design(cache['z'], target_error, legal, weights)
            arrays = {k: v for k, v in cache.items() if isinstance(v, np.ndarray)}
            arrays.update(design_error=target_error, design_legal=legal, design_weights=weights, theta_parent=theta)
            cache_pin = self.save_arrays(f'cache-{seed}.npz', arrays)
            design_seconds = (self.clock.now_ns() - tick) / 1e9
            cache_meta = {k: v for k, v in cache.items() if not isinstance(v, np.ndarray)}
            cache_meta['helper_perf_counter_seconds'] = helper_seconds
            cached_losses = {}
            U = solver.contrast_basis()
            for family in ('pretrained', 'action_residual_only'):
                increment = self.theta(states[family, seed]) - theta
                coefficients = (U.T @ increment).reshape(-1)
                cached_losses[family] = float(np.square(A @ coefficients - b).sum())
            rows = []
            for family in ARMS:
                self.check()
                self.receipt['pending'] = {'seed': seed, 'stage': 'solve', 'family': family}
                tick = self.clock.now_ns()
                result = solver.solve_design(A, b, mode=family)
                seconds = (self.clock.now_ns() - tick) / 1e9
                self.receipt['solves_completed'] += 1
                self.check()
                tick = self.clock.now_ns()
                head = (theta + result['increment']).astype(np.float32)
                state = {k: v.clone() for k, v in parent.items()}
                state['action_residual.weight'] = torch.from_numpy(head[:, :-1].copy())
                state['action_residual.bias'] = torch.from_numpy(head[:, -1].copy())
                require(all(torch.equal(state[k], parent[k]) for k in frozen), 'export preserves backbone')
                pin = self.save_arrays(f'checkpoint-{family}-{seed}.npz',
                                       {'slow.' + k: v.numpy().copy() for k, v in state.items()})
                export_seconds = (self.clock.now_ns() - tick) / 1e9
                states[family, seed] = state
                self.receipt['fits_completed'] += 1
                exported_coef = (U.T @ (self.theta(state) - theta)).reshape(-1)
                exported_loss = float(np.square(A @ exported_coef - b).sum())
                row = {'family': family, 'seed': seed, 'cache': cache_pin, 'cache_metadata': cache_meta,
                       'cache_design_seconds': design_seconds, 'solve_seconds': seconds,
                       'export_seconds': export_seconds, 'checkpoint': pin, 'frozen_names': frozen,
                       'cached_control_losses': cached_losses, 'exported_cached_loss': exported_loss,
                       'solver': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}}
                rows.append(row)
                self.receipt['pending'] = None
            del A, b
            for family in TRAIN_VIEWS:
                row, predicted = self.predict(family, seed, states[family, seed], history, 'train')
                expected = cache_module.affine_prediction(cache, self.theta(states[family, seed]))
                actual = predicted['action_prediction'].astype(np.float64) / 64
                actual[history['prior_mask']] = predicted['corrected_shadow_prior'][history['prior_mask']].astype(np.float64) / 64
                support = cache['support_mask']
                difference = np.abs(actual[support] - expected[support])
                tolerance = CONFIG['parity_atol'] + CONFIG['parity_rtol'] * np.abs(expected[support])
                parity = {'passed': bool((difference <= tolerance).all()),
                          'max_abs': float(difference.max(initial=0)),
                          'max_tolerance_ratio': float((difference / tolerance).max(initial=0)),
                          'supported_rows': int(support.sum())}
                row['cache_parity'] = parity
                views.append(row)
                require(parity['passed'], 'ordinary TRAIN export agrees with declared cached affine surrogate')
                if family in ARMS:
                    fit = next(r for r in rows if r['family'] == family)
                    fit['train_validation_seconds'] = row['seconds']
                    fit['standalone_adaptation_seconds'] = (cache['seconds'] + design_seconds + fit['solve_seconds']
                        + fit['export_seconds'] + row['seconds'])
                    fit['parity'] = parity
            fits.extend(rows)
            print(json.dumps({'seed': seed, 'solves': self.receipt['solves_completed'],
                              'train_views': self.receipt['train_views_completed']}), flush=True)
        fit_pin = self.publish('fits.json', fits)
        self.publish('training-barrier.json', {'solves': 6, 'train_views': 12, 'dev_decodes': 0,
            'fits': fit_pin, 'created_ns': self.clock.now_ns(), 'checkpoints': [f['checkpoint'] for f in fits]})
        dev = self.history('dev')
        for seed in SEEDS:
            for family in VIEWS:
                row, _ = self.predict(family, seed, states[family, seed], dev, 'dev')
                views.append(row)
        self.publish('views.json', views)
        self.publish('summary.json', {'status': 'completed_pending_independent_audit', 'solves': 6,
            'train_views': 12, 'dev_views': 15, 'dev_reused': True, 'fresh_dev': False})

    def predict(self, family, seed, state, history, stage):
        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_query_memory_model as models
        np, torch = self.np, self.torch
        tick = self.clock.now_ns()
        model = models.from_states(models.memory.Config('none', key_dim=8), seed, 4, state, None, slow_mode='frozen')
        model.eval()
        offsets = history['episode_offsets']
        total = int(offsets[-1])
        saved = {k: np.zeros((total, 4), np.float32) for k in ('action_prediction', 'corrected_shadow_prior')}
        saved.update(prior_mask=np.zeros(total, np.bool_), episode_offsets=offsets.copy())
        work = {}
        with torch.no_grad():
            for index, (low, high) in enumerate(pairwise(offsets)):
                low, high = int(low), int(high)
                carry = model.initial_carry(1)
                for start in range(0, high - low, 32):
                    self.check()
                    self.receipt['pending'] = {'view': family, 'seed': seed, 'episode': index, 'start': start}
                    packet = data.batch_chunk(history, [index], start)
                    inputs = {k: torch.from_numpy(v) for k, v in packet['model_inputs'].items()}
                    forecast = model(**inputs, carry=carry)
                    length = int(inputs['lengths'][0])
                    for name in ('action_prediction', 'corrected_shadow_prior', 'prior_mask'):
                        value = getattr(forecast, name)[0]
                        require(value[length:].detach().numpy().tobytes() == np.zeros_like(value[length:].detach().numpy()).tobytes(), 'positive zero padding')
                        saved[name][low + start:low + start + length] = value[:length].detach().numpy()
                    carry = models.detach_carry(forecast.carry)
                    for k, v in forecast.work_counts.items():
                        work[k] = work.get(k, 0) + v
                    self.receipt['pending'] = None
                require(bool(carry.slow.base.ended.all()) and int(carry.slow.base.absolute_step[0]) == high - low, 'complete DEV episode')
        require(saved['action_prediction'][history['query_mask']].tobytes() == history['query_scores'][history['query_mask']].tobytes(), 'exact observed query scores')
        require(np.array_equal(saved['prior_mask'], history['prior_mask']), 'exact later-query support')
        episodes = []
        for i, identity in enumerate(self.plan[stage]['identities']):
            low, high = map(int, offsets[i:i + 2])
            episodes.append(metrics.episode_metrics(identity, 4, history['targets'][low:high], history['legal'][low:high],
                saved['action_prediction'][low:high], prequery_forecast=saved['corrected_shadow_prior'][low:high]))
        report = metrics.aggregate_episodes(episodes, family=family, fit_seed=seed, expected_identities=self.plan[stage]['identities'])
        pin = self.save_arrays(f'{stage}-prediction-{family}-{seed}.npz', saved)
        row = {'stage': stage, 'family': family, 'seed': seed, 'metrics': report, 'prediction': pin,
               'seconds': (self.clock.now_ns() - tick) / 1e9, 'work': work}
        self.event({'event': 'view_complete', **row}, 'progress.jsonl')
        self.receipt['train_views_completed' if stage == 'train' else 'views_completed'] += 1
        return row, saved


    def audit(self):
        plan, receipt, directory = process_closure(self.args.plan, self.args.producer_receipt, self.args.producer_terminal)
        require(receipt['phase'] == 'train' and read(self.args.producer_terminal)['finished_ns'] <= self.launch['started_ns'],
                'audit after original successful producer')
        self.receipt['extra_arguments'] = ['--producer-receipt', str(self.args.producer_receipt),
                                           '--producer-terminal', str(self.args.producer_terminal)]
        self.receipt['producer_receipt'] = descriptor(self.args.producer_receipt)
        self.receipt['producer_terminal'] = descriptor(self.args.producer_terminal)
        import numpy as np
        audit = load('scripts/audit_otto_direct_readout.py', '_direct_independent_audit')
        result = audit.audit(np, plan, receipt, directory, self.check)
        self.publish('audit.json', result)
        self.receipt['audit_counts'] = result['counts']
        for key in ('array_decodes', 'checkpoint_decodes', 'views_completed', 'train_views_completed'):
            self.receipt[key] = result['counts'][key]
        self.receipt['agreement'] = result['agreement']

    def finish(self, error=None):
        if not self.owns_output:
            return
        if error is None:
            require(descriptor(self.args.plan)['sha256'] == self.args.plan_sha256, 'plan unchanged')
            validate_plan(self.plan)
            verify(self.receipt['supervision'])
            self.check()
            require({p.name for p in self.out.iterdir()} == expected_payloads(self.phase), 'complete output roster')
        try:
            finished = self.clock.now_ns() if self.clock else None
        except BaseException as clock_error:
            if error is None:
                raise
            self.receipt['clock_error'] = repr(clock_error)
            finished = None
        self.receipt.update(status='completed' if error is None else 'failed', error=error,
            finished_ns=finished, wall_seconds=(finished - self.start) / 1e9 if finished is not None else None,
            requires_original_supervisor_closure=True,
            files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
        write(self.out / 'receipt.json', self.receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--engineering', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    for phase in LIMITS:
        p = sub.add_parser(phase)
        for name in ('plan', 'supervision', 'output'):
            p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--plan-sha256', required=True)
        if phase == 'audit':
            p.add_argument('--producer-receipt', type=Path, required=True)
            p.add_argument('--producer-terminal', type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.output.is_absolute() and args.output.is_relative_to(ROOT)
            and '..' not in args.output.parts and not args.output.exists()
            and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'exclusive contained output')
    if args.mode == 'plan':
        freeze(args)
        return
    run = BoundRun(args)
    def interrupted(_signum, _frame):
        raise TimeoutError('original supervisor termination')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run.initialize()
        run.bind()
        run.train() if args.mode == 'train' else run.audit()
        run.finish()
    except BaseException:
        run.finish(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
