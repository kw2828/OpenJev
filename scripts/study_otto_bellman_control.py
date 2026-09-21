"""Paired MC and observation-backup continuation of authenticated MLPs.

The original study is immutable. This runner reuses its public episode mechanics,
not its training lifecycle or aggregate. A separate TRAIN-only qualification
exercises the real continuation path before any full-study execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OLD = 'scripts/study_otto_return_value.py'
OLD_PIN = '3afdf21506dc60db9a91a18200dfe005133ab1b9398f3baa12614498253abc11'
VERSION = 'otto-bellman-control-v1'
SEEDS = (10101, 10102, 10103)
FAMILIES = ('mc', 'backup', 'reference')
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in FAMILIES) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
HORIZON, CASES = 2188, 48
FIRST = {'lambda3': 12100001, 'lambda4': 12200001, 'lambda5': 12300001}
TEMPLATE_FIRST = 12500001
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
PROTOCOL = 'research/otto-bellman-control-protocol.md'
NEW_SOURCES = (OLD, 'scripts/study_otto_bellman_control.py',
               'src/openjev/research/otto_bellman_learning.py',
               'src/openjev/research/otto_bellman_targets.py',
               'scripts/audit_otto_bellman_targets.py',
               'tests/test_otto_bellman_learning.py',
               'tests/test_otto_bellman_targets.py',
               'tests/test_audit_otto_bellman_targets.py',
               'tests/test_otto_bellman_control.py', PROTOCOL)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


require(sha(ROOT / OLD) == OLD_PIN, 'unchanged inherited numerical and episode source')
P = load(ROOT / OLD, '_bellman_previous')
read, write, descriptor, closed, regular = P.read, P.write, P.descriptor, P.closed, P.regular


def limits(mode):
    if mode == 'qualify':
        return {'native_seconds': 180, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2,
                'native_steps': 0, 'native_resets': 0, 'optimizer_updates': 24, 'target_readouts': 128}
    require(mode == 'study', 'known mode')
    return {'native_seconds': 10800, 'rss_bytes': 8 * 1024**3, 'output_bytes': 8 * 1024**3,
            'native_steps': 1440 * HORIZON, 'native_resets': 1443,
            'optimizer_updates': 10560, 'target_readouts': 134136}


def configuration(mode):
    require(mode in ('qualify', 'study'), 'known configuration mode')
    return {'training_rows': 64 if mode == 'qualify' else 5589,
            'training_seed_order': [10101] if mode == 'qualify' else list(SEEDS),
            'fit_mode_order': ['mc', 'backup'], 'epochs': 6 if mode == 'qualify' else 40,
            'batch_size': 32 if mode == 'qualify' else 128, 'learning_rate': .001,
            'gradient_clip': 5., 'refresh_every': 5, 'shuffle_seed_offset': 30000,
            'checkpoint': 'fixed final', 'parity_rows': list(range(16)),
            'parity_models': 3 if mode == 'qualify' else 9,
            'diagnostic_forward_rows': 0 if mode == 'qualify' else 40188,
            'qualification_rows': 'numpy.linspace(0,5588,64,dtype=int64)',
            'horizon': HORIZON, 'evaluation_cases_per_regime': 48 if mode == 'study' else 0,
            'evaluation_arms': list(ARMS), 'initial_hit': '1+(case%6)//2', 'block': 'case//6',
            'amortization_searches': [1, 100, 10000], 'external_model_calls': 0}


def payload_names(mode):
    names = {'started.json', 'runtime.json', 'preparation.json', 'work-contexts.jsonl', 'work.jsonl',
             'initializations.jsonl', 'updates.jsonl', 'epoch-orders.jsonl', 'fit-curves.jsonl', 'fits.jsonl',
             'target-checkpoints.jsonl', 'target-refreshes.jsonl', 'parity.jsonl', 'target-audit.json', 'training-costs.json'}
    seeds = (10101,) if mode == 'qualify' else SEEDS
    for seed in seeds:
        names.add(f'mc-targets-{seed}.npz')
        names.update(f'{prefix}-{arm}-{seed}.npz' for prefix in ('initial', 'final') for arm in ('mc', 'backup'))
        for index in range(2 if mode == 'qualify' else 8):
            names.add(f'target-checkpoint-backup-{seed}-{index:02d}.npz')
            names.add(f'targets-backup-{seed}-{index:02d}.npz')
        if mode == 'study':
            names.update(f'predictions-{arm}-{seed}.npz' for arm in ('mc', 'backup'))
    if mode == 'study':
        names.update({'prediction-diagnostics.jsonl', 'native-setup.json', 'inference-setup.json',
                      'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json'})
    return names


def evaluation_order(first):
    require(set(first) == set(REGIMES), 'three evaluation seed ranges')
    for ri, regime in enumerate(REGIMES):
        for case in range(CASES):
            offset = (ri * CASES + case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield regime, first[regime] + case, case // 6, 1 + (case % 6) // 2, arm


def successful_terminal(worker_path, plan_path, terminal_path):
    worker, plan, terminal = read(worker_path), read(plan_path), read(terminal_path)
    started = read(worker_path.parent / 'started.json')
    request, launch = started['request'], started['launch']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True, 'successful original process terminal')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
        require(terminal[key] == launch[key], 'original process/worker launch join')
    require(sha(Path(request['supervision'])) == worker['supervision_sha256']
            and read(Path(request['supervision'])) == launch, 'launch artifact identity')
    require(request == {'plan': str(plan_path), 'plan_sha256': sha(plan_path),
                        'output': str(worker_path.parent), 'supervision': request['supervision']}, 'exact original request')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command[:2] == [plan['python_executable'], str(ROOT / 'scripts/study_otto_bellman_control.py')]
            and len(command) == 10
            and dict(zip(command[2::2], command[3::2], strict=True)) == {f'--{k.replace("_", "-")}': v for k, v in request.items()},
            'exact qualified command')
    require(launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and worker['started_ns'] == started['started_ns']
            and launch['deadline_ns'] == launch['started_ns'] + plan['limits']['native_seconds'] * 10**9
            and launch['cap_seconds'] == plan['limits']['native_seconds']
            and worker['wall_seconds'] == (worker['finished_ns'] - worker['started_ns']) / 1e9
            and terminal['elapsed_ns'] == terminal['finished_ns'] - terminal['started_ns']
            and worker['clock_backend'] == launch['clock_backend'], 'bounded original qualification timing')
    require(launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
            and launch['clock_source_sha256'] == P.CLOCK_PIN and launch['cwd'] == str(ROOT)
            and launch['pid'] == launch['pgid'] != launch['parent_pid'], 'qualification isolated supervisor identity')


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink()
            and sha(args.plan) == args.plan_sha256, 'external plan hash')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_execution'
            and plan['mode'] in ('qualify', 'study') and plan['limits'] == limits(plan['mode']), 'frozen mode and limits')
    require(plan['configuration'] == configuration(plan['mode']), 'fixed recipe and cohort configuration')
    require(plan['evaluation_first_seeds'] == FIRST and plan['template_first_seed'] == TEMPLATE_FIRST, 'fixed fresh seeds')
    require(set(NEW_SOURCES) <= plan['sources'].keys() and plan['sources'][OLD] == OLD_PIN, 'new source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'source hash: {name}')
    roles = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt'}
    require(set(plan['inputs']) == roles, 'exact input roles')
    paths = {}
    for role, value in plan['inputs'].items():
        paths[role] = regular(value['path'])
        require(descriptor(paths[role]) == {k: value[k] for k in ('sha256', 'bytes')}, 'input descriptor')
    previous = P.authenticate(SimpleNamespace(plan=paths['prior_plan'], plan_sha256=sha(paths['prior_plan'])))
    require(all(plan['sources'].get(k) == v for k, v in previous['sources'].items()), 'prior source closure unchanged')
    worker, audit = closed(paths['prior_receipt']), closed(paths['prior_audit_receipt'])
    terminal = read(paths['prior_terminal'])
    require(set(worker['files']) == P.payload_names() and worker['version'] == previous['version']
            and worker['plan_sha256'] == sha(paths['prior_plan']) and worker['sources'] == previous['sources']
            and worker['inputs'] == previous['inputs'] and worker['completed_fits'] == 9
            and worker['completed_episodes'] == 720 and worker['pending'] == []
            and worker['external_model_calls'] == 0
            and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'completed prior scalar study')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and not terminal['timed_out'] and terminal['group_absent']
            and terminal['cleanup']['reaped'] and not terminal['cleanup']['errors']
            and terminal['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'],
            'prior original process terminal')
    require(audit['agreement'] is True and audit['plan_sha256'] == sha(paths['prior_plan'])
            and audit['worker_sha256'] == sha(paths['prior_receipt'])
            and audit['terminal_sha256'] == sha(paths['prior_terminal']), 'independently audited prior inputs')
    require(sys.version.split()[0] == plan['python_version'] and sys.executable == plan['python_executable'], 'interpreter')
    require({d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == plan['all_distributions'], 'runtime closure')
    if plan['mode'] == 'study':
        require(set(plan['qualification']) == {'plan', 'receipt', 'terminal'}, 'full qualification identity')
        qpaths = {}
        for role, desc in plan['qualification'].items():
            qpaths[role] = regular(desc['path'])
            require(descriptor(qpaths[role]) == {k: desc[k] for k in ('sha256', 'bytes')}, 'qualification artifact pin')
        qplan = authenticate(SimpleNamespace(plan=qpaths['plan'], plan_sha256=sha(qpaths['plan'])))
        receipt = closed(qpaths['receipt'])
        require(qplan['mode'] == receipt['mode'] == 'qualify' and receipt['completed_fits'] == 2
                and receipt['target_audit_passed'] and receipt['parity_passed']
                and receipt['pending'] == [] and receipt['plan_sha256'] == sha(qpaths['plan'])
                and receipt['sources'] == qplan['sources'] and receipt['inputs'] == qplan['inputs']
                and set(receipt['files']) == payload_names('qualify')
                and all(v['attempted'] == v['returned'] for v in receipt['calls'].values()), 'completed real pipeline qualification')
        successful_terminal(qpaths['receipt'], qpaths['plan'], qpaths['terminal'])
        require(all(plan['sources'].get(k) == v for k, v in receipt['sources'].items()), 'qualified sources unchanged')
    return plan


def summarize(rows, mixtures, first):
    require([(r['regime'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows]
            == list(evaluation_order(first)), 'complete ordered 1440 episodes')
    for row in rows:
        require(type(row['found']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= HORIZON
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0 and row['final_update_assimilated'], 'complete capped outcomes')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in P.METRICS)
                and abs(row['controller_seconds'] - math.fsum(row[k] for k in P.TIMES)) <= 1e-9, 'finite complete controller costs')
    regimes, competence, improvement = {}, [], []
    def add(group, name, value, threshold, passed):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passed)})
    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(0 < v < 1 for v in weights.values())
                and abs(math.fsum(weights.values()) - 1) <= 1e-12, 'fixed positive mixture')
        selected = [r for r in rows if r['regime'] == regime]
        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h] * math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))
        means = {a: {m: weighted([r for r in selected if r['arm'] == a], m) for m in P.METRICS} for a in ARMS}
        blocks = [{a: weighted([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS) / 3 for m in P.METRICS} for f in FAMILIES}
        teacher, candidate = means['analytic_inbounds'], family['backup']
        for seed in SEEDS:
            v = means[f'backup@{seed}']
            add(competence, f'{regime}.{seed}.success', v['found'], .95, v['found'] >= .95)
            add(competence, f'{regime}.{seed}.moves', v['steps'], 1.05 * teacher['steps'], v['steps'] <= 1.05 * teacher['steps'])
        for control in ('mc', 'reference'):
            ref = family[control]
            positive = sum(math.fsum(b[f'{control}@{s}'] - b[f'backup@{s}'] for s in SEEDS) / 3 > 0 for b in blocks)
            add(improvement, f'{regime}.{control}.success', candidate['found'], ref['found'], candidate['found'] >= ref['found'])
            add(improvement, f'{regime}.{control}.moves', candidate['steps'], .95 * ref['steps'], candidate['steps'] <= .95 * ref['steps'])
            add(improvement, f'{regime}.{control}.positive_blocks', positive, 6, positive >= 6)
            add(improvement, f'{regime}.{control}.controller_cost', candidate['controller_seconds'], ref['controller_seconds'], candidate['controller_seconds'] <= ref['controller_seconds'])
        regimes[regime] = {'weights': weights, 'means': means, 'family_means': family, 'blocks': blocks,
                          'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h) / 16
                                                 for m in P.METRICS} for a in ARMS} for h in (1, 2, 3)},
                          'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 48} for a in ARMS}}
    require((len(competence), len(improvement)) == (18, 24), 'all42 continuation checks')
    return {'version': VERSION, 'episodes': len(rows), 'paired_cases': 144, 'regimes': regimes,
            'competence_checks': competence, 'improvement_checks': improvement,
            'pilot_continuation': all(v['passes'] for v in competence + improvement),
            'comparison': 'Equal optimizer updates; target construction is additional measured work.',
            'learned_architecture_advantage_established': False}


class Run:
    emit, sync = P.Run.emit, P.Run.sync
    restore, physical = P.Run.restore, P.Run.physical
    public, witness, episode = P.Run.public, P.Run.witness, P.Run.episode

    def __init__(self, args):
        self.args, self.out = args, args.output
        self.plan = self.launch = self.clock = self.start = None
        self.handles, self.calls, self.pending, self.last, self.context_ids = {}, {}, [], {}, {}
        self.sequence, self.io_seconds = 0, 0.
        self.context = {'phase': 'bind'}
        self.receipt = {'version': VERSION, 'status': 'started', 'external_model_calls': 0,
                        'completed_fits': 0, 'completed_episodes': 0, 'target_refreshes': 0,
                        'target_audit_passed': False, 'parity_passed': False}

    def check(self):
        if self.launch:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared deadline')
        if self.plan is None:
            return
        caps = self.plan['limits']
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= caps['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= caps['output_bytes'], 'output cap')
        for channel, key in (('native_step', 'native_steps'), ('native_reset', 'native_resets'),
                             ('optimizer_update', 'optimizer_updates'), ('target_readout', 'target_readouts')):
            require(self.calls.get(channel, {}).get('attempted', 0) <= caps[key], f'{channel} cap')

    def begin(self, channel, *, check=True):
        if check:
            self.check()
        caps = {'native_step': 'native_steps', 'native_reset': 'native_resets',
                'optimizer_update': 'optimizer_updates', 'target_readout': 'target_readouts'}
        record = self.calls.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        if channel in caps:
            require(record['attempted'] < self.plan['limits'][caps[channel]], 'work allocation before invocation')
        record['attempted'] += 1
        self.sequence += 1
        call_id = self.sequence
        self.pending.append({'id': call_id, 'channel': channel, 'context': dict(self.context)})
        context = {k: v for k, v in self.context.items() if k != 'step'}
        key = json.dumps(context, sort_keys=True)
        if key not in self.context_ids:
            self.context_ids[key] = len(self.context_ids)
            self.emit('work-contexts.jsonl', {'id': self.context_ids[key], 'context': context})
        self.emit('work.jsonl', [call_id, 0, channel, self.context_ids[key], self.context.get('step')])
        return call_id, time.perf_counter(), self.io_seconds

    def end(self, channel, handle):
        call_id, tick, io = handle
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        duration = raw - excluded
        require(duration >= 0 and self.pending[-1]['id'] == call_id, 'ordered nonnegative accounting')
        self.pending.pop()
        self.calls[channel]['returned'] += 1
        self.calls[channel]['seconds'] += duration
        self.last[channel] = {'seconds': duration, 'instrumented_seconds': raw, 'excluded_io_seconds': excluded}
        self.emit('work.jsonl', [call_id, 1, channel, duration, raw, excluded])

    def call(self, channel, operation, *, check=True):
        handle = self.begin(channel, check=check)
        result = operation()
        self.end(channel, handle)
        return result

    def bind(self):
        require(sha(ROOT / P.CLOCK) == P.CLOCK_PIN, 'clock source')
        self.clock = load(ROOT / P.CLOCK, '_bellman_clock').SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, 'supervision missing')
            time.sleep(.01)
        self.launch, self.plan = read(self.args.supervision), authenticate(self.args)
        launch = self.launch
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch['pid'] == os.getpid()
                and launch['pgid'] == os.getpgrp() and launch['parent_pid'] == os.getppid()
                and launch['cwd'] == str(ROOT) == str(Path.cwd()), 'actual supervised process identity')
        require(launch['clock_backend'] == self.clock.backend and launch['started_ns'] <= self.start < launch['deadline_ns']
                and launch['cap_seconds'] == self.plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns'] + launch['cap_seconds'] * 10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == P.CLOCK_PIN, 'shared supervisor cap')
        self.receipt.update(mode=self.plan['mode'], limits=self.plan['limits'], sources=self.plan['sources'],
                            inputs=self.plan['inputs'], plan_sha256=self.args.plan_sha256,
                            supervision_sha256=sha(self.args.supervision))
        write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'launch': launch, 'started_ns': self.start})

    def setup(self):
        os.environ.update(THREADS)
        tick = time.perf_counter()
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.np, self.torch = np, torch
        sys.path.insert(0, str(ROOT / 'src'))
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        self.SourceTracking = load(ROOT / P.UPSTREAM, '_bellman_native').SourceTracking
        model_tick = time.perf_counter()
        from openjev.research import otto_return_value, otto_value_branches
        self.model, self.branches = otto_return_value, otto_value_branches
        self.model_module_setup_seconds = time.perf_counter() - model_tick
        from openjev.research import otto_bellman_learning, otto_bellman_targets
        self.learning, self.targets = otto_bellman_learning, otto_bellman_targets
        self.shared_setup_seconds = time.perf_counter() - tick - self.model_module_setup_seconds
        self.prior_run = regular(self.plan['inputs']['prior_receipt']['path']).parent
        self.kernels, self.mixtures = {}, {}
        for regime in REGIMES:
            with np.load(self.prior_run / f'kernel-{regime}.npz', allow_pickle=False) as archive:
                kernel, weights = archive['likelihood'], archive['initial_hit_weights']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64
                    and weights.shape == (4,) and weights[0] == 0, 'fixed public kernel')
            kernel.setflags(write=False)
            self.kernels[regime] = kernel
            self.mixtures[regime] = {h: float(weights[h]) for h in (1, 2, 3)}
        if self.plan['mode'] == 'study':
            checks = []
            for i, regime in enumerate(REGIMES):
                self.context = {'phase': 'native_setup', 'regime': regime}
                env = self.environment(regime, self.plan['template_first_seed'] + i, None)
                weights = np.asarray(next(r for r in env.draw_log if r['channel'] == 'initial')['probabilities'])
                require(np.array_equal(env.p_Poisson, self.kernels[regime])
                        and all(float(weights[h]) == self.mixtures[regime][h] for h in (1, 2, 3)), 'native cached kernel/mixture parity')
                checks.append({'regime': regime, 'template_seed': self.plan['template_first_seed'] + i,
                               'cached_kernel_exact': True, 'initial_public': self.public(env, 0)})
            write(self.out / 'native-setup.json', {'checks': checks, 'native_resets': 3})
        write(self.out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
              'environment': THREADS, 'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'shared_setup_seconds': self.shared_setup_seconds, 'model_module_setup_seconds': self.model_module_setup_seconds,
              'module_allocation_episodes': 1296, 'module_scope': 'Canonical value model and explicit branch imports; excludes training-only setup'})

    def environment(self, regime, seed, initial_hit):
        config = {'Ndim': 2, 'lambda_over_dx': REGIMES[regime], 'R_dt': 2., 'Ngrid': 53,
                  'Nhits': 4, 'draw_source': True, 'norm_Poisson': 'Euclidean'}
        return self.call('native_reset', lambda: self.seeded(self.SourceTracking, seed, config, initial_hit=initial_hit))

    def prepare(self):
        np = self.np
        tick = time.perf_counter()
        with np.load(self.prior_run / 'train-data.npz', allow_pickle=False) as archive:
            require(set(archive.files) == {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'exact TRAIN array schema')
            full = {k: archive[k] for k in archive.files}
        metadata = [json.loads(line) for line in (self.prior_run / 'train-rows.jsonl').read_text().splitlines()]
        expected = {'features': ((5589, 11028), np.float32), 'target': ((5589,), np.float32),
                    'beliefs': ((5589, 53, 53), np.float64), 'positions': ((5589, 2), np.int64),
                    'sensing_length': ((5589,), np.float64)}
        for key, (shape, dtype) in expected.items():
            require(full[key].shape == shape and full[key].dtype == dtype and np.isfinite(full[key]).all(), 'TRAIN shape/dtype')
        require(len(metadata) == 5589 and len({r['episode_id'] for r in metadata}) == 192, 'complete TRAIN identity')
        for i, row in enumerate(metadata):
            require(row['row_index'] == i and row['stage'] == 'train' and not row['public']['done']
                    and row['target'] == float(full['target'][i])
                    and row['target'] == (row['total_steps'] - row['prefix_index']) / 64
                    and row['posterior']['sha256'] == hashlib.sha256(full['beliefs'][i].tobytes()).hexdigest(), 'exact TRAIN row join')
        indices = np.linspace(0, 5588, 64, dtype=np.int64) if self.plan['mode'] == 'qualify' else np.arange(5589, dtype=np.int64)
        data = {k: v[indices].copy() for k, v in full.items()}
        data['metadata'] = [metadata[i] for i in indices]
        data['row_indices'] = indices
        self.training_data, self.full_training = data, full
        self.full_metadata = metadata
        self.full_allowed = [r['public']['valid_actions'] for r in metadata]
        self.preparation_seconds = time.perf_counter() - tick
        write(self.out / 'preparation.json', {'input_rows': 5589, 'input_episodes': 192,
            'selected_rows': len(indices), 'row_indices': indices.tolist(), 'mode': self.plan['mode'],
            'seconds': self.preparation_seconds, 'selection': 'linspace(0,5588,64) qualification; all rows study',
            'input_files': {name: descriptor(self.prior_run / name) for name in ('train-data.npz', 'train-rows.jsonl')}})
        return data

    def save_npz(self, name, arrays):
        tick = time.perf_counter()
        path = self.out / name
        with path.open('xb') as stream:
            self.np.savez_compressed(stream, **arrays)
        self.io_seconds += time.perf_counter() - tick
        return descriptor(path)

    def fit(self, data):
        np = self.np
        qualification = self.plan['mode'] == 'qualify'
        recipe = self.learning.Recipe(epochs=6, batch_size=32) if qualification else self.learning.Recipe()
        self.refresh_records, self.fit_records = [], []
        self.valid_data = None
        if not qualification:
            with np.load(self.prior_run / 'valid-data.npz', allow_pickle=False) as archive:
                self.valid_data = {k: archive[k] for k in archive.files}
        for seed in ((10101,) if qualification else SEEDS):
            initial_path = self.prior_run / f'final-mlp8-{seed}.npz'
            initial = self.restore(initial_path)
            self.parity(f'reference@{seed}', initial, seed)
            for mode in ('mc', 'backup'):
                self.fit_one(data, initial_path, initial, seed, mode, recipe, qualification)
        self.audit_targets(data)
        self.receipt['parity_passed'] = True
        write(self.out / 'training-costs.json', {'preparation_seconds': self.preparation_seconds,
            'fits': {r['fit_id']: r['fit_seconds'] for r in self.fit_records},
            'scope': 'Fit wall includes restoration, optimization, target checkpoints/branches/readouts/storage and journals. '
                     'Separate parity, diagnostics and independent target audit are excluded from fit wall and retained in operation counts.'})

    def fit_one(self, data, initial_path, initial, seed, mode, recipe, qualification):
        np = self.np
        fit_id = f'{mode}@{seed}'
        self.context = {'phase': 'fit', 'fit_id': fit_id}
        operations, checkpoints, target_payloads = {}, {}, {}
        start = time.perf_counter()
        def hook(event, payload):
            if event in ('operation_attempt', 'operation_return'):
                context = payload['progress']['context']
                self.context = {'fit_id': fit_id, **context}
                operation = payload['operation']
                if event == 'operation_attempt':
                    operations[operation] = self.begin(operation)
                else:
                    self.end(operation, operations.pop(operation))
                    if operation == 'optimizer_update':
                        self.emit('updates.jsonl', {'fit_id': fit_id, **context, **payload['result']})
            elif event == 'target_checkpoint':
                index = payload['refresh_index']
                name = f'target-checkpoint-{mode}-{seed}-{index:02d}.npz'
                desc = self.save_npz(name, payload['checkpoint'])
                checkpoints[index] = {'path': name, **desc, 'content_identity': payload['checkpoint_identity']}
                self.emit('target-checkpoints.jsonl', {'fit_id': fit_id, 'refresh_index': index,
                          'epoch': payload['epoch'], **checkpoints[index]})
            elif event == 'initial':
                desc = self.save_npz(f'initial-{mode}-{seed}.npz', payload['checkpoint'])
                self.emit('initializations.jsonl', {'fit_id': fit_id, 'initial_source': str(initial_path.relative_to(ROOT)),
                    'initial_source_sha256': sha(initial_path), 'saved': desc,
                    **{k: v for k, v in payload.items() if k != 'checkpoint'}})
            elif event == 'targets':
                self.save_npz(f'mc-targets-{seed}.npz', {k: payload[k] for k in ('targets_float64', 'targets_float32')})
            elif event == 'refresh':
                index = payload['refresh_index']
                saved = target_payloads.pop(index)
                require(np.array_equal(payload['targets_float64'], saved['targets_float64'])
                        and np.array_equal(payload['targets_float32'], saved['targets_float32']), 'training target cast matches materialized targets')
                record = {k: v for k, v in payload.items() if k not in ('checkpoint', 'targets_float64', 'targets_float32')}
                record.update(fit_id=fit_id, checkpoint=checkpoints[index],
                              payload=self.refresh_records[-1]['payload'])
                self.emit('target-refreshes.jsonl', record)
                self.receipt['target_refreshes'] += 1
            elif event == 'order':
                self.emit('epoch-orders.jsonl', {'fit_id': fit_id, **{k: v for k, v in payload.items() if k != 'order'},
                          'order': payload['order'].tolist()})
            elif event == 'epoch':
                self.emit('fit-curves.jsonl', {'fit_id': fit_id, **payload})
                if payload['epoch'] % 5 == 0 or payload['epoch'] == recipe.epochs:
                    print(json.dumps({'phase': 'fit', 'fit_id': fit_id, 'epoch': payload['epoch'],
                                      'mse': payload['training_mse_normalized']}), flush=True)
            elif event == 'final':
                self.save_npz(f'final-{mode}-{seed}.npz', payload['checkpoint'])
            elif event == 'failure':
                self.emit('learning-failures.jsonl', {'fit_id': fit_id, **payload})

        def target_builder(frozen, metadata):
            index = metadata['refresh_index']
            require(index in checkpoints and checkpoints[index]['content_identity'] == metadata['checkpoint_identity'],
                    'persisted exact target checkpoint before readout')
            pieces = {name: [] for name in self.targets.BellmanTargets.__slots__}
            readout_handle = None
            for offset in range(0, len(data['target']), 32):
                self.check()
                stop = min(offset + 32, len(data['target']))
                def observer(event, local_index, rows, offset=offset):
                    nonlocal readout_handle
                    require(rows == 16, 'exact target readout width')
                    self.context = {'phase': 'target_generation', 'fit_id': fit_id, 'refresh_index': index,
                                    'step': int(data['row_indices'][offset + local_index])}
                    if event == 'attempt':
                        require(readout_handle is None, 'no overlapping target readouts')
                        readout_handle = self.begin('target_readout', check=False)
                    else:
                        self.end('target_readout', readout_handle)
                        readout_handle = None
                batch = self.targets.build_targets(frozen, data['beliefs'][offset:stop], data['positions'][offset:stop],
                    [self.kernels[data['metadata'][i]['regime']] for i in range(offset, stop)],
                    data['sensing_length'][offset:stop],
                    [data['metadata'][i]['public']['valid_actions'] for i in range(offset, stop)], observe=observer)
                for name, parts in pieces.items():
                    parts.append(getattr(batch, name))
            payload = {name: np.concatenate(parts) for name, parts in pieces.items()}
            payload['row_indices'] = data['row_indices'].copy()
            name = f'targets-{mode}-{seed}-{index:02d}.npz'
            desc = self.save_npz(name, payload)
            self.refresh_records.append({'fit_id': fit_id, 'refresh_index': index,
                'checkpoint': checkpoints[index], 'payload': {'path': name, **desc}})
            target_payloads[index] = {k: payload[k] for k in ('targets_float64', 'targets_float32')}
            return payload['targets_float64']

        result = self.learning.continue_learning(initial, data['features'], data['target'].astype(np.float64),
            seed=seed, mode=mode, recipe=recipe, target_builder=target_builder if mode == 'backup' else None, hook=hook)
        require(not operations and not target_payloads, 'closed training callbacks')
        duration = time.perf_counter() - start
        record = {'fit_id': fit_id, 'seed': seed, 'mode': mode, 'rows': len(data['target']),
            'epochs': recipe.epochs, 'optimizer_steps': list(result.optimizer_steps), 'fit_seconds': duration,
            'initial_identity': result.initial_identity, 'final_identity': result.final_identity,
            'checkpoint': f'final-{mode}-{seed}.npz', 'checkpoint_sha256': sha(self.out / f'final-{mode}-{seed}.npz'),
            'progress': result.progress, 'refreshes': list(result.refreshes)}
        self.fit_records.append(record)
        self.emit('fits.jsonl', record)
        self.receipt['completed_fits'] += 1
        self.parity(fit_id, result.final_export, seed)
        if not qualification:
            self.predict_final(fit_id, result.final_export)
        self.sync()

    def parity(self, fit_id, exported, seed):
        np, torch = self.np, self.torch
        self.context = {'phase': 'parity_restore', 'fit_id': fit_id}
        reference = self.call('parity_restore', lambda: self.learning.restore_mlp(exported, seed).double().eval())
        head = self.model.FrozenValue(exported)
        for index in range(16):
            self.context = {'phase': 'parity', 'fit_id': fit_id, 'step': index}
            data = self.full_training
            b, pos, lam = data['beliefs'][index], data['positions'][index], float(data['sensing_length'][index])
            x = self.model.value_features(P.center(b[None], pos[None], np), pos[None], lam)
            def torch_value(features):
                with torch.no_grad():
                    return reference(torch.from_numpy(features)).numpy()
            scalar_np = self.call('parity_numpy_forward', lambda x=x: head.normalized(x))
            scalar_ref = self.call('parity_torch_forward', lambda x=x, torch_value=torch_value: torch_value(x))
            branch = self.branches.rl_branches(b, pos, self.kernels[f'lambda{int(lam)}'], self.full_allowed[index])
            saved = {}
            def values(z, positions, kernel, kind, lam=lam, saved=saved, torch_value=torch_value):
                features = self.model.value_features(z, positions, lam)
                result = self.call(f'parity_{kind}_forward', lambda: 64 * (head.normalized(features) if kind == 'numpy' else torch_value(features)))
                saved[kind] = result
                return result
            costs_np = self.branches.explicit_scores(branch, lambda z, p, k, values=values: values(z, p, k, 'numpy'), arithmetic='float64')
            costs_ref = self.branches.explicit_scores(branch, lambda z, p, k, values=values: values(z, p, k, 'torch'), arithmetic='float64')
            action_np = self.branches.select_action(costs_np, branch.eligible_actions)
            action_ref = self.branches.select_action(costs_ref, branch.eligible_actions)
            close = lambda a, b: bool(np.all(np.abs(a - b) <= 1e-10 + 1e-10 * np.abs(b)))
            passed = close(scalar_np, scalar_ref) and close(saved['numpy'], saved['torch']) and close(costs_np, costs_ref) and action_np == action_ref
            self.emit('parity.jsonl', {'fit_id': fit_id, 'row_index': index, 'scalar_numpy': scalar_np.tolist(),
                'scalar_torch': scalar_ref.tolist(), 'values_numpy': saved['numpy'].tolist(), 'values_torch': saved['torch'].tolist(),
                'costs_numpy': costs_np.tolist(), 'costs_torch': costs_ref.tolist(), 'allowed_actions': list(branch.eligible_actions),
                'action_numpy': action_np, 'action_torch': action_ref, 'passed': passed})
            require(passed, 'fixed final deployment parity')

    def predict_final(self, fit_id, exported):
        np = self.np
        head = self.model.FrozenValue(exported)
        predictions = {}
        stats = {}
        for split, data in (('train', self.full_training), ('valid', self.valid_data)):
            values = []
            for offset in range(0, len(data['target']), 256):
                self.context = {'phase': 'saved_predictions', 'fit_id': fit_id, 'split': split, 'step': offset}
                stop = min(offset + 256, len(data['target']))
                # Regime-homogeneous contiguous groups retain each row's sensing scalar.
                part = []
                for i in range(offset, stop):
                    features = self.model.value_features(P.center(data['beliefs'][i:i+1], data['positions'][i:i+1], np),
                        data['positions'][i:i+1], float(data['sensing_length'][i]))
                    part.append(self.call('saved_value_forward', lambda features=features: head.normalized(features), check=False))
                values.append(np.concatenate(part))
            prediction = np.concatenate(values)
            predictions[split] = prediction
            error = prediction - data['target'].astype(np.float64)
            stats[split] = {'rows': len(error), 'mse_normalized': float(np.mean(error**2)),
                            'mae_physical': float(64 * np.mean(np.abs(error)))}
        self.save_npz(f'predictions-{fit_id.replace("@", "-")}.npz', predictions)
        self.emit('prediction-diagnostics.jsonl', {'fit_id': fit_id, 'statistics': stats,
                  'scope': 'Fixed final exported checkpoint against original MC labels; no selection.'})

    def audit_targets(self, data):
        auditor = load(ROOT / 'scripts/audit_otto_bellman_targets.py', '_bellman_independent_targets')
        expected = data['row_indices']
        audits = []
        for record in self.refresh_records:
            self.context = {'phase': 'target_audit', 'fit_id': record['fit_id'], 'refresh_index': record['refresh_index']}
            path = self.out / record['payload']['path']
            checkpoint_path = self.out / record['checkpoint']['path']
            require(descriptor(path) == {k: record['payload'][k] for k in ('sha256', 'bytes')}, 'saved target payload unchanged')
            require(descriptor(checkpoint_path) == {k: record['checkpoint'][k] for k in ('sha256', 'bytes')}, 'saved target checkpoint unchanged')
            payload, checkpoint = self.restore(path), self.restore(checkpoint_path)
            result = self.call('target_audit', lambda payload=payload, checkpoint=checkpoint, checkpoint_path=checkpoint_path, record=record: auditor.audit_refresh(payload, checkpoint,
                train=self.full_training, eligible_actions=self.full_allowed,
                kernels={REGIMES[k]: v for k, v in self.kernels.items()}, expected_row_indices=expected,
                checkpoint_sha256=sha(checkpoint_path), expected_checkpoint_sha256=record['checkpoint']['sha256'],
                c0=float(checkpoint['c0']), check=self.check))
            audits.append({'fit_id': record['fit_id'], 'refresh_index': record['refresh_index'], **result})
        write(self.out / 'target-audit.json', {'status': 'completed', 'refreshes': audits})
        self.receipt['target_audit_passed'] = True
        self.receipt['target_audit_readouts'] = sum(r['audit_checkpoint_readout_calls'] for r in audits)


    def evaluate(self):
        heads, setup = {}, {}
        for arm in ARMS[:-1]:
            family, seed = arm.split('@')
            path = (self.prior_run / f'final-mlp8-{seed}.npz') if family == 'reference' else (self.out / f'final-{family}-{seed}.npz')
            self.context = {'phase': 'inference_setup', 'arm': arm}
            tick = time.perf_counter()
            heads[arm] = self.model.FrozenValue(self.restore(path))
            setup[arm] = {'seconds': time.perf_counter() - tick, 'checkpoint': str(path.relative_to(ROOT)),
                          'sha256': sha(path), 'allocated_episodes': 144, 'storage': heads[arm].storage_bytes()}
        write(self.out / 'inference-setup.json', setup)
        rows, sources, uniforms = [], {}, {}
        for regime, seed, block, hit, arm in evaluation_order(self.plan['evaluation_first_seeds']):
            row = self.episode(regime, seed, hit, arm, block, heads.get(arm))
            allocation = (setup[arm]['seconds'] / 144 + self.model_module_setup_seconds / 1296) if arm in setup else 0.
            row['setup_allocation_seconds'] = allocation
            row['controller_seconds'] += allocation
            self.emit('evaluation.jsonl', row)
            key = (regime, seed)
            source = row['source_evaluation_only']
            require(key not in sources or sources[key] == source, 'matched sampled source')
            sources[key] = source
            for draw in row['draws_evaluation_only']:
                index = (*key, draw['channel'], draw['index'])
                require(index not in uniforms or uniforms[index] == draw['uniform'], 'matched categorical uniform')
                uniforms[index] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            if len(rows) % 10 == 0:
                print(json.dumps({'phase': 'eval', 'episodes': len(rows), 'native_steps': self.calls['native_step']['returned']}), flush=True)
        result = summarize(rows, self.mixtures, self.plan['evaluation_first_seeds'])
        result['amortization'] = self.amortization(result, setup)
        result.update(inference_setup=setup, shared_setup_seconds=self.shared_setup_seconds,
                      model_module_setup_seconds=self.model_module_setup_seconds, calls=self.calls,
                      paired_source_cases=len(sources), paired_uniforms=len(uniforms))
        write(self.out / 'summary.json', result)
        return result

    def amortization(self, result, setup):
        fits = {row['fit_id']: row['fit_seconds'] for row in self.fit_records}
        scenarios = {}
        for regime, panel in result['regimes'].items():
            scenarios[regime] = {}
            for arm, metrics in panel['means'].items():
                if arm == 'analytic_inbounds':
                    learn, preparation, deployment, utility = 0., 0., 0., metrics['controller_seconds']
                else:
                    learn = fits.get(arm, 0.)
                    preparation = self.preparation_seconds / 6 if arm in fits else 0.
                    deployment = setup[arm]['seconds'] + self.model_module_setup_seconds / 9
                    utility = metrics['controller_seconds'] - metrics['setup_allocation_seconds']
                scenarios[regime][arm] = {'continuation_seconds': learn, 'preparation_share_seconds': preparation,
                    'deployment_setup_seconds': deployment, 'controller_without_deployment_seconds': utility,
                    'seconds_per_search': {str(h): utility + (learn + preparation + deployment) / h for h in (1, 100, 10000)}}
        return {'scope': 'Incremental continuation plus declared deployment accounting; original training is a common prior investment. '
                         'No new searches or compute-matched claim. Analytic actor initialization is paid anew each search and included in U.',
                'original_training_costs': read(self.prior_run / 'training-costs.json'),
                'original_runtime': read(self.prior_run / 'runtime.json'), 'scenarios': scenarios}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists(), 'exclusive absolute output')
        self.out.mkdir(parents=True, exist_ok=False)
        primary = None
        try:
            self.bind()
            self.setup()
            data = self.prepare()
            self.fit(data)
            result = self.evaluate() if self.plan['mode'] == 'study' else {'pilot_continuation': False}
            self.sync()
            require({p.name for p in self.out.iterdir() if p.is_file()} == payload_names(self.plan['mode']), 'exact completed output closure')
            require(authenticate(self.args) == self.plan and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged execution closure')
            require(not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values()), 'no unfinished work')
            for key, channel in (('optimizer_updates', 'optimizer_update'), ('target_readouts', 'target_readout'), ('native_resets', 'native_reset')):
                require(self.calls.get(channel, {}).get('returned', 0) == self.plan['limits'][key], 'complete fixed work')
            models = 3 if self.plan['mode'] == 'qualify' else 9
            for channel in ('parity_numpy_forward', 'parity_torch_forward'):
                require(self.calls[channel]['returned'] == models * 32, 'complete fixed parity work')
            require(self.calls['parity_restore']['returned'] == models
                    and self.receipt['target_audit_readouts'] == self.plan['limits']['target_readouts']
                    and self.calls.get('saved_value_forward', {}).get('returned', 0) == self.plan['configuration']['diagnostic_forward_rows'],
                    'complete independent target audit and diagnostic work')
            require(self.receipt['completed_fits'] == (2 if self.plan['mode'] == 'qualify' else 6)
                    and self.receipt['completed_episodes'] == (0 if self.plan['mode'] == 'qualify' else 1440)
                    and self.receipt['parity_passed'] and self.receipt['target_audit_passed'], 'complete fits, audit, parity and evaluation')
            self.receipt.update(status='completed', pilot_continuation=result['pilot_continuation'])
            self.check()
        except BaseException as error:
            primary = error
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            errors = []
            try:
                self.sync()
                for stream in self.handles.values():
                    stream.close()
            except BaseException as error:  # noqa: BLE001 - Preserve journal failures and the original exception.
                errors.append(f'Journal finalization: {error!r}')
            try:
                end = self.clock.now_ns() if self.clock else None
                self.receipt.update(calls=self.calls, pending=self.pending, context=self.context,
                    started_ns=self.start, finished_ns=end, clock_backend=self.clock.backend if self.clock else None,
                    wall_seconds=(end - self.start) / 1e9 if end is not None and self.start is not None else None,
                    artifact_io_seconds=self.io_seconds,
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                self.check()
            except BaseException as error:  # noqa: BLE001 - A cap or clock failure still needs a failure receipt.
                errors.append(f'Final clock/hash/limit check: {error!r}')
                self.receipt.setdefault('started_ns', self.start)
                self.receipt.setdefault('finished_ns', None)
                self.receipt.setdefault('wall_seconds', None)
            if errors:
                self.receipt.update(status='failed', finalization_errors=errors,
                                    calls=self.calls, pending=self.pending, context=self.context)
            try:
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as error:  # noqa: BLE001 - Do not replace a primary execution failure.
                errors.append(f'Receipt publication: {error!r}')
            if errors:
                if primary is not None:
                    primary.add_note('; '.join(errors))
                else:
                    raise RuntimeError('; '.join(errors))
        try:
            self.check()
        except BaseException as error:
            try:
                write(self.out / 'late-failure.json', {'status': 'failed', 'error': repr(error)})
                (self.out / 'receipt.json').rename(self.out / 'completion-before-late-failure.json')
                self.receipt.update(status='failed', error=f'Late publication failure: {error!r}',
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                write(self.out / 'receipt.json', self.receipt)
            except OSError as secondary:
                error.add_note(f'Late failure preservation: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'supervision', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    Run(parser.parse_args()).execute()
