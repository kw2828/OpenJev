"""Fixed full-belief action-head pilot with one shared student-data round.

Only public posteriors and the supplied sensor model enter learned features.
TRAIN exposure is pooled across all six collectors; final epochs are fixed.
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

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-symmetry-head-v1'
MODEL = 'src/openjev/research/otto_symmetry_head.py'
PROTOCOL = 'research/otto-symmetry-head-protocol.md'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
UPSTREAM = 'tmp/otto-source-review-01/isotropic/classes/sourcetracking.py'
FAMILIES, SEEDS = ('shared', 'dense'), (9101, 9102, 9103)
KINDS = {'shared': 'd4_shared', 'dense': 'dense_augmented'}
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in ('shared', 'dense', 'dense_ensemble')) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'train': {'lambda3': 910001, 'lambda4': 920001},
         'valid': {'lambda3': 930001, 'lambda4': 940001},
         'dagger': {'lambda3': 950001, 'lambda4': 960001},
         'eval': {'lambda3': 970001, 'lambda4': 980001, 'lambda5': 990001}}
HORIZON, CASES, EPISODES = 2188, 24, 720
LIMITS = {'native_seconds': 5400, 'rss_bytes': 8 * 1024**3, 'output_bytes': 2 * 1024**3,
          'native_steps': 1104 * HORIZON + 256, 'native_resets': 1115, 'optimizer_updates': 63360}
CONFIGURATION = {'families': list(FAMILIES), 'fit_seeds': list(SEEDS), 'arms': list(ARMS), 'regimes': REGIMES,
                 'first_seeds': FIRST, 'train_cases_per_regime': 96, 'validation_cases_per_regime': 24,
                 'dagger_cases_per_fit_per_regime': 12, 'evaluation_cases_per_regime': CASES,
                 'horizon': HORIZON, 'prefix_cap_per_episode': 64, 'epochs_per_stage': 40,
                 'batch_size': 128, 'learning_rate': .0003, 'gradient_norm_clip': 5., 'teacher_temperature': .25,
                 'checkpoint_rule': 'fixed final epoch40 in each stage; validation descriptive only',
                 'optimizer_continued': True, 'training_weighting': 'equal total weight per retained episode',
                 'qualification_first_seed': 1000001, 'qualification_cases': 8, 'qualification_horizon': 32,
                 'setup_template_resets': 3, 'Ngrid': 53, 'Nhits': 4, 'R_dt': 2., 'Ndim': 2,
                 'norm_Poisson': 'Euclidean', 'episodes': EPISODES, 'training_updates_are_new': True}
ROLES = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt'}
REQUIRED = {MODEL, PROTOCOL, CLOCK, UPSTREAM, 'src/openjev/research/otto_public.py',
            'src/openjev/research/otto_released_policy.py', 'src/openjev/research/otto_reference_control.py',
            'scripts/study_otto_symmetry_head.py', 'tests/test_otto_symmetry_study.py',
            'tests/test_otto_symmetry_head.py', 'scripts/audit_otto_symmetry_head.py',
            'tests/test_audit_otto_symmetry_head.py',
            'scripts/supervise_dialogue_observation_v2.py'}
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
TIMES = ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')
METRICS = ('steps', 'found', *TIMES, 'controller_seconds', 'environment_seconds', 'state_bytes')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def regular(name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, 'contained relative input')
    p = ROOT / p
    require(p.is_file() and not any(v.is_symlink() for v in (p, *p.parents)), 'regular nonsymlink input')
    return p


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def closed(path):
    receipt = read(path)
    require(receipt['status'] == 'completed', 'completed input receipt')
    files = receipt['files']
    actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file()}
    require(actual == set(files) | {path.name}, 'exact closed payload membership')
    for name, desc in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'payload containment')
        p = path.parent / rel
        require(not any(v.is_symlink() for v in (p, *p.parents)) and descriptor(p) == desc, 'closed payload hash')
    return receipt


def authenticate(args):
    require(args.plan.is_absolute() and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256, 'external plan hash')
    plan = read(args.plan)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_native_run'
            and plan['configuration'] == CONFIGURATION and plan['limits'] == LIMITS, 'frozen scientific configuration')
    require(REQUIRED <= plan['sources'].keys() and plan['sources'][CLOCK] == CLOCK_PIN, 'complete source closure')
    for name, pin in plan['sources'].items():
        require(sha(regular(name)) == pin, f'source hash: {name}')
    require(set(plan['inputs']) == ROLES, 'exact prior boundary input roles')
    paths = {}
    for role, value in plan['inputs'].items():
        paths[role] = regular(value['path'])
        require(descriptor(paths[role]) == {k: value[k] for k in ('sha256', 'bytes')}, 'external prior input descriptor')
    prior = read(paths['prior_plan'])
    require(prior['version'] == 'otto-boundary-control-v1' and prior['status'] == 'frozen_before_native_run'
            and all(plan['sources'].get(k) == v for k, v in prior['sources'].items()), 'unchanged inherited boundary source closure')
    for value in prior['inputs'].values():
        require(descriptor(regular(value['path'])) == {k: value[k] for k in ('sha256', 'bytes')}, 'unchanged inherited input')
    worker, audit = closed(paths['prior_receipt']), closed(paths['prior_audit_receipt'])
    terminal, started = read(paths['prior_terminal']), read(paths['prior_receipt'].parent / 'started.json')
    require(worker['version'] == prior['version'] and worker['sources'] == prior['sources'] and worker['inputs'] == prior['inputs']
            and worker['plan_sha256'] == sha(paths['prior_plan']) and worker['completed_episodes'] == 768
            and worker['work']['pending'] == [] and worker['paired_neural_prefixes'] == 192, 'complete prior study identity')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
            and terminal['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < terminal['deadline_ns'],
            'successful prior parent terminal')
    require(sha(Path(started['request']['supervision'])) == worker['supervision_sha256'], 'prior launch pin')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend'):
        require(terminal[key] == started['launch'][key], 'prior parent/worker launch identity')
    require(audit['version'] == 'otto-boundary-control-saved-audit-v1' and audit['agreement'] is True
            and audit['plan_sha256'] == sha(paths['prior_plan']) and audit['worker_sha256'] == sha(paths['prior_receipt'])
            and audit['terminal_sha256'] == sha(paths['prior_terminal'])
            and audit['source']['sha256'] == plan['sources']['scripts/audit_otto_boundary_control.py']
            and audit['model_calls'] == audit['simulator_calls'] == audit['policy_calls'] == 0, 'completed independent prior audit')
    require(sys.version.split()[0] == plan['python_version'] and sys.executable == plan['python_executable'], 'runtime interpreter identity')
    require(plan['runtime_versions'] == {'numpy': '2.5.3', 'scipy': '1.18.1', 'torch': '2.14.0'}, 'fixed numerical versions')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'installed runtime version')
    require({d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == plan['all_distributions'],
            'complete installed distribution closure')
    return plan


def sample_indices(length):
    require(type(length) is int and 1 <= length <= HORIZON, 'complete episode decision length')
    count = min(64, length)
    return [0] if count == 1 else [round(i * (length - 1) / (count - 1)) for i in range(count)]


def row_weights(metadata):
    counts = {}
    for row in metadata:
        counts[row['episode_id']] = counts.get(row['episode_id'], 0) + 1
    require(bool(counts), 'nonempty complete episodes')
    return [len(metadata) / (len(counts) * counts[r['episode_id']]) for r in metadata]


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST['eval'].items()):
        for case in range(CASES):
            offset = (ri * CASES + case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield regime, first + case, case // 3, 1 + case % 3, arm


def choose(scores, allowed, np):
    mask = np.asarray([a in allowed for a in range(4)], dtype=bool)
    require(scores.shape == (4,) and np.isfinite(scores[mask]).all() and mask.any(), 'finite permitted costs')
    best = scores[mask].min()
    return int(np.flatnonzero(mask & (np.abs(scores - best) < 1e-10))[0])


def teacher_target(scores, allowed, np):
    mask = np.asarray([a in allowed for a in range(4)], dtype=bool)
    values = np.asarray(scores, dtype=np.float64)[mask]
    require(values.size and np.isfinite(values).all(), 'finite analytic labels')
    logits = -(values - values.min()) / max(float(values.max() - values.min()), 1e-8) / .25
    weights = np.exp(logits)
    target = np.zeros(4, dtype=np.float32)
    target[mask] = weights / weights.sum()
    return target, mask


def summary(rows, mixtures):
    expected = list(evaluation_order())
    require(len(rows) == EPISODES and [(r['regime'], r['seed'], r['block'], r['initial_hit'], r['arm']) for r in rows] == expected,
            'exact complete720 episode cohort')
    for row in rows:
        require(type(row['steps']) is int and 1 <= row['steps'] <= HORIZON and type(row['found']) is bool
                and (row['found'] or row['steps'] == HORIZON) and row['updates'] == row['steps']
                and row['blocked_steps'] == 0, 'complete final updates and censored outcomes')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in METRICS)
                and abs(row['controller_seconds'] - math.fsum(row[k] for k in TIMES)) <= 1e-9, 'complete finite cost')
    regimes, competence, compression, architecture = {}, [], [], []

    def check(group, name, value, threshold, passes):
        group.append({'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passes)})

    for regime in REGIMES:
        weights = {int(k): v for k, v in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(0 < v < 1 for v in weights.values())
                and abs(math.fsum(weights.values()) - 1) <= 1e-12, 'positive normalized mixture')
        selected = [r for r in rows if r['regime'] == regime]

        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h] * math.fsum(float(r[metric]) for r in subset if r['initial_hit'] == h)
                             / sum(r['initial_hit'] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: weighted([r for r in selected if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: weighted([r for r in selected if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS) / 3 for m in METRICS}
                  for f in ('shared', 'dense', 'dense_ensemble')}
        teacher, candidate = means['analytic_inbounds'], family['shared']
        for seed in SEEDS:
            value = means[f'shared@{seed}']
            check(competence, f'{regime}.{seed}.success', value['found'], .95, value['found'] >= .95)
            check(competence, f'{regime}.{seed}.moves', value['steps'], 1.05 * teacher['steps'], value['steps'] <= 1.05 * teacher['steps'])
        check(compression, f'{regime}.success', candidate['found'], teacher['found'], candidate['found'] >= teacher['found'])
        check(compression, f'{regime}.moves', candidate['steps'], 1.05 * teacher['steps'], candidate['steps'] <= 1.05 * teacher['steps'])
        check(compression, f'{regime}.cost80', candidate['controller_seconds'], .8 * teacher['controller_seconds'], candidate['controller_seconds'] <= .8 * teacher['controller_seconds'])
        worst = max(means[f'shared@{s}']['controller_seconds'] for s in SEEDS)
        check(compression, f'{regime}.every_cost', worst, teacher['controller_seconds'], worst < teacher['controller_seconds'])
        for control in ('dense', 'dense_ensemble'):
            ref = family[control]
            positive = sum(math.fsum(b[f'{control}@{s}'] - b[f'shared@{s}'] for s in SEEDS) / 3 > 0 for b in blocks)
            check(architecture, f'{regime}.{control}.success', candidate['found'], ref['found'], candidate['found'] >= ref['found'])
            check(architecture, f'{regime}.{control}.moves', candidate['steps'], .95 * ref['steps'], candidate['steps'] <= .95 * ref['steps'])
            check(architecture, f'{regime}.{control}.positive_blocks', positive, 6, positive >= 6)
            check(architecture, f'{regime}.{control}.cost', candidate['controller_seconds'], ref['controller_seconds'], candidate['controller_seconds'] <= ref['controller_seconds'])
        regimes[regime] = {'weights': weights, 'means': means, 'family_means': family, 'blocks': blocks,
                          'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in selected if r['arm'] == a and r['initial_hit'] == h) / 8
                                                 for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                          'raw_counts': {a: {'found': sum(r['found'] for r in selected if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    require((len(competence), len(compression), len(architecture)) == (18, 12, 24), 'all54 conditions')
    return {'version': VERSION, 'episodes': len(rows), 'regimes': regimes,
            'competence_checks': competence, 'compression_checks': compression, 'architecture_checks': architecture,
            'pilot_continuation': all(c['passes'] for c in competence + compression + architecture),
            'learned_architecture_advantage_established': False, 'inherited_gate_revised': False}


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = self.plan = None
        self.handles, self.calls, self.pending = {}, {}, []
        self.sequence, self.io_seconds = 0, 0.
        self.context_ids = {}
        self.context = {'phase': 'setup'}
        self.last = {}
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS, 'external_model_calls': 0,
                        'completed_stage_fits': 0, 'completed_episodes': 0, 'collection_episodes': 0, 'training_updates': 0}

    def check(self):
        if self.launch:
            require(self.clock.now_ns() < self.launch['deadline_ns'], 'shared native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'output cap')
        for channel, cap in (('native_step', LIMITS['native_steps']), ('native_reset', LIMITS['native_resets'])):
            require(self.calls.get(channel, {}).get('attempted', 0) <= cap, 'native work cap')
        require(self.calls.get('optimizer_update', {}).get('attempted', 0) <= LIMITS['optimizer_updates'], 'optimizer update cap')

    def emit(self, name, value):
        tick = time.perf_counter()
        if name not in self.handles:
            self.handles[name] = (self.out / name).open('x')
        stream = self.handles[name]
        stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()  # Survives worker termination; fsync at each closed episode.
        self.io_seconds += time.perf_counter() - tick

    def call(self, channel, operation, *, check=True):
        if check:
            self.check()
        record = self.calls.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        caps = {'native_step': LIMITS['native_steps'], 'native_reset': LIMITS['native_resets'],
                'optimizer_update': LIMITS['optimizer_updates']}
        require(channel not in caps or record['attempted'] < caps[channel], 'work allocation before invocation')
        record['attempted'] += 1
        self.sequence += 1
        call_id = self.sequence
        self.pending.append({'id': call_id, 'channel': channel, 'context': dict(self.context)})
        # Save episode/phase identity once, retaining the current step per call.
        context = {k: v for k, v in self.context.items() if k != 'step'}
        context_key = json.dumps(context, sort_keys=True)
        if context_key not in self.context_ids:
            context_id = len(self.context_ids)
            self.context_ids[context_key] = context_id
            self.emit('work-contexts.jsonl', {'id': context_id, 'context': context})
        self.emit('work.jsonl', [call_id, 0, channel, self.context_ids[context_key], self.context.get('step')])
        tick, io = time.perf_counter(), self.io_seconds
        value = operation()
        raw, excluded = time.perf_counter() - tick, self.io_seconds - io
        duration = raw - excluded
        require(duration >= 0, 'nonnegative measured operation cost')
        record['returned'] += 1
        record['seconds'] += duration
        require(self.pending[-1]['id'] == call_id, 'nested operation accounting')
        self.pending.pop()
        self.last[channel] = {'seconds': duration, 'instrumented_seconds': raw, 'excluded_io_seconds': excluded}
        self.emit('work.jsonl', [call_id, 1, channel, duration, raw, excluded])
        return value

    def sync(self):
        for stream in self.handles.values():
            stream.flush()
            os.fsync(stream.fileno())

    def bind(self):
        require(sha(ROOT / CLOCK) == CLOCK_PIN, 'clock source before import')
        self.clock = load(ROOT / CLOCK, '_symmetry_clock').SuspendClock()
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
                and launch['cap_seconds'] == LIMITS['native_seconds']
                and launch['deadline_ns'] == launch['started_ns'] + LIMITS['native_seconds'] * 10**9
                and launch['watchdog_sha256'] == self.plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                and launch['clock_source_sha256'] == CLOCK_PIN, 'actual shared supervisor cap')
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=sha(self.args.supervision),
                            sources=self.plan['sources'], inputs=self.plan['inputs'])
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
        self.SourceTracking = load(ROOT / UPSTREAM, '_symmetry_source').SourceTracking
        model_tick = time.perf_counter()
        self.model = load(ROOT / MODEL, '_symmetry_head_model')
        self.model_module_setup_seconds = time.perf_counter() - model_tick
        self.shared_setup_seconds = time.perf_counter() - tick - self.model_module_setup_seconds
        self.kernels, self.mixtures = {}, {}
        # Reach the old qualified kernels through each externally pinned input.
        def inherited(owner, role):
            value = owner['inputs'][role]
            path = regular(value['path'])
            require(descriptor(path) == {k: value[k] for k in ('sha256', 'bytes')}, 'inherited kernel lineage')
            return path
        boundary = read(inherited(self.plan, 'prior_plan'))
        reference = read(inherited(boundary, 'prior_plan'))
        native = read(inherited(reference, 'qualification_plan'))
        old_kernels = {'lambda3': inherited(native, 'base_kernel'), 'lambda4': inherited(native, 'shift_kernel')}
        from scipy.special import k0
        from scipy.stats import poisson
        setup_checks = []
        for i, (name, lam) in enumerate(REGIMES.items()):
            env = self.environment(name, 1000101 + i, None)
            kernel = env.p_Poisson.copy()
            weights = self.np.asarray(next(r for r in env.draw_log if r['channel'] == 'initial')['probabilities'])
            require(env.N == 53 and env.Nhits == 4 and kernel.shape == (4, 107, 107)
                    and weights.shape == (4,) and weights[0] == 0 and np.all(weights[1:] > 0), 'fixed native geometry and mixture')
            if name in old_kernels:
                with np.load(old_kernels[name], allow_pickle=False) as archive:
                    require(np.array_equal(kernel, archive['likelihood']) and np.array_equal(weights, archive['initial_hit_weights']),
                            'exact previously qualified kernel and initial mixture')
            cells = np.indices((107, 107)) - 53
            distance = np.sqrt(np.sum(cells**2, axis=0))
            mu = 2 * k0(np.where(distance == 0, 1, distance) / lam) / np.log(2 * lam)
            theory = np.stack((poisson.pmf(0, mu), poisson.pmf(1, mu), poisson.pmf(2, mu), poisson.sf(2, mu)))
            theory[:, 53, 53] = 0
            error = float(np.max(np.abs(kernel - theory)))
            require(error <= 1e-12, 'known radial native likelihood formula')
            setup_checks.append({'regime': name, 'template_seed': 1000101 + i, 'maximum_kernel_formula_error': error,
                                 'old_kernel_exact': name in old_kernels, 'source_evaluation_only': env.source.tolist(),
                                 'initial_public': self.public(env, 0)})
            kernel.setflags(write=False)
            self.kernels[name] = kernel
            self.mixtures[name] = {h: float(weights[h]) for h in (1, 2, 3)}
            np.savez_compressed(self.out / f'kernel-{name}.npz', likelihood=kernel, initial_hit_weights=weights)
        write(self.out / 'native-setup.json', {'checks': setup_checks, 'native_resets': 3})
        write(self.out / 'runtime.json', {'python': sys.version, 'executable': sys.executable,
              'versions': {n: importlib.metadata.version(n) for n in self.plan['runtime_versions']},
              'environment': THREADS, 'torch_threads': torch.get_num_threads(),
              'torch_interop_threads': torch.get_num_interop_threads(), 'shared_setup_seconds': self.shared_setup_seconds,
              'model_module_setup_seconds': self.model_module_setup_seconds, 'module_allocation_episodes': 648})

    def environment(self, regime, seed, initial_hit):
        config = {'Ndim': 2, 'lambda_over_dx': REGIMES[regime], 'R_dt': 2., 'Ngrid': 53,
                  'Nhits': 4, 'draw_source': True, 'norm_Poisson': 'Euclidean'}
        return self.call('native_reset', lambda: self.seeded(self.SourceTracking, seed, config, initial_hit=initial_hit))

    def public(self, env, step):
        return dict(self.observation(env, step)._asdict())

    def witness(self, actor, env, public):
        belief = actor.belief
        require(actor.public == public and belief.shape == (53, 53) and belief.dtype == self.np.float64
                and belief.tobytes() == env.p_source.tobytes() and self.np.isfinite(belief).all(), 'exact native/public belief')
        return {'sha256': hashlib.sha256(belief.tobytes()).hexdigest(), 'mass': float(belief.sum())}

    def qualify(self):
        checks = []
        before = self.calls.get('native_step', {}).get('returned', 0)
        for case in range(8):
            self.context = {'phase': 'qualification', 'case': case}
            env = self.environment('lambda5', 1000001 + case, 1 + case % 3)
            current = self.public(env, 0)
            actor = self.actor_class(current, self.kernels['lambda5'], allow_stay=False)
            self.witness(actor, env, current)
            self.emit('qualification.jsonl', {'kind': 'reset', 'case': case, 'public': current, 'source_evaluation_only': env.source.tolist()})
            for step in range(1, 33):
                action = (0, 2, 1, 3)[(step - 1) % 4]
                self.context['step'] = step
                self.call('native_step', lambda action=action, env=env: env.step(action, quiet=True))
                after = self.public(env, step)
                actor.update(action, after)
                state = self.witness(actor, env, after)
                self.emit('qualification.jsonl', {'kind': 'step', 'case': case, 'step': step, 'action': action, 'public': after, 'posterior': state})
                if after['done']:
                    break
            checks.append({'case': case, 'steps': step, 'found': after['done'], 'passed': True})
        count = self.calls['native_step']['returned'] - before
        require(count <= 256 and len(checks) == 8, 'fixed bounded kernel5 qualification')
        write(self.out / 'qualification.json', {'status': 'completed', 'checks': checks, 'native_steps': count, 'resets': 8,
              'scope': 'Public filtering at prescribed fresh mechanical prefixes; no neural policy or efficacy evidence.'})
        self.sync()

    def episode(self, stage, regime, seed, hit, behavior, *, block=None, head=None, ensemble=False):
        np = self.np
        episode_id = f'{stage}:{regime}:{seed}:{behavior}'
        self.context = {'phase': stage, 'episode': episode_id, 'step': 0}
        env = self.environment(regime, seed, hit)
        current = self.public(env, 0)
        tick = time.perf_counter()
        actor = self.actor_class(current, self.kernels[regime], allow_stay=False)
        fmap = self.model.PublicFeatureMap(self.kernels[regime], REGIMES[regime]) if head is not None or stage != 'eval' else None
        init_seconds = time.perf_counter() - tick
        state = self.witness(actor, env, current)
        filename = 'eval-transitions.jsonl' if stage == 'eval' else 'collection-transitions.jsonl'
        identity = {'episode_id': episode_id, 'stage': stage, 'regime': regime, 'seed': seed, 'initial_hit': hit, 'arm': behavior, 'block': block}
        self.emit(filename, {'kind': 'reset', **identity, 'public': current, 'posterior_after': state, 'source_evaluation_only': env.source.tolist()})
        retained, choices, updates, native = [], 0., 0., 0.
        raw_choices = excluded_io = 0.
        for step in range(1, HORIZON + 1):
            self.context['step'] = step
            self.check()  # Resource/filesystem monitoring is outside controller cost.
            feature = target = mask = teacher_costs = None
            tick, io = time.perf_counter(), self.io_seconds
            if stage != 'eval':
                _, teacher_costs = self.call('teacher_label', actor._policy._value_policy, check=False)
                target, mask = teacher_target(teacher_costs, current['valid_actions'], np)
            if head is None:
                if teacher_costs is None:
                    _, costs = self.call('analytic_choose', actor._policy._value_policy, check=False)
                else:
                    costs = teacher_costs
                action = choose(costs, current['valid_actions'], np)
                if stage != 'eval':
                    feature = fmap.features(actor.belief, current)
            else:
                feature = fmap.features(actor.belief, current)
                costs = self.call('head_predict', lambda feature=feature: head.scores(feature, kind='dense_ensemble' if ensemble else None), check=False)
                action = choose(costs, current['valid_actions'], np)
            raw = time.perf_counter() - tick
            excluded = self.io_seconds - io
            choice_seconds = raw - excluded
            choices += choice_seconds
            raw_choices += raw
            excluded_io += excluded
            if stage != 'eval':
                retained.append((feature, target, mask, {'episode_id': episode_id, 'stage': stage, 'regime': regime, 'seed': seed,
                    'initial_hit': hit, 'arm': behavior, 'prefix_index': step - 1, 'public': current, 'posterior': state,
                    'teacher_costs': [float(v) if np.isfinite(v) else None for v in teacher_costs]}))
            result = self.call('native_step', lambda action=action: env.step(action, quiet=True))
            native_cost = self.last['native_step']['seconds']
            native += native_cost
            after = self.public(env, step)
            require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'returned native event')
            tick = time.perf_counter()
            actor.update(action, after)
            elapsed = time.perf_counter() - tick
            updates += elapsed
            after_state = self.witness(actor, env, after)
            self.emit(filename, {'kind': 'step', 'episode_id': episode_id, 'step': step, 'action': action,
                'costs': [float(v) if np.isfinite(v) else None for v in costs], 'allowed_actions': list(current['valid_actions']),
                'public': after, 'posterior_before': state, 'posterior_after': after_state,
                'choose_seconds': choice_seconds, 'choose_instrumented_seconds': raw, 'choose_excluded_io_seconds': excluded,
                'update_seconds': elapsed, 'environment_seconds': native_cost, 'native_p_end': float(result[1])})
            require(current['position'] != after['position'], 'inbounds actions always move')
            current, state = after, after_state
            if current['done']:
                break
        draws = [{k: r[k] for k in ('channel', 'index', 'uniform', 'selected_index', 'cdf_mass')} for r in env.draw_log]
        row = {**identity, 'steps': step, 'found': current['done'], 'updates': step, 'blocked_steps': 0,
               'init_seconds': init_seconds, 'choose_seconds': choices, 'update_seconds': updates,
               'setup_allocation_seconds': 0., 'controller_seconds': init_seconds + choices + updates,
               'choose_instrumented_seconds': raw_choices, 'choose_excluded_io_seconds': excluded_io,
               'environment_seconds': native, 'state_bytes': actor.storage_bytes()['mutable_array_bytes'],
               'storage': {'public_actor': actor.storage_bytes(),
                           'feature_kernel_bytes': int(fmap.kernel.nbytes) if fmap is not None else 0,
                           'head': head.storage_bytes() if head is not None else None},
               'source_evaluation_only': env.source.tolist(), 'draws_evaluation_only': draws, 'final_public': current,
               'final_update_assimilated': True}
        self.emit('eval-episodes.jsonl' if stage == 'eval' else 'collection-episodes.jsonl', row)
        selected = [retained[i] for i in sample_indices(step)] if retained else []
        if stage != 'eval':
            self.receipt['collection_episodes'] += 1
        self.sync()
        return row, selected

    def dataset(self, stage, heads=None):
        records = []
        count = {'train': 96, 'valid': 24, 'dagger': 12}[stage]
        behaviors = [(f'{f}@{s}', heads[f'{f}@{s}']) for s in SEEDS for f in FAMILIES] if stage == 'dagger' else [('teacher', None)]
        for regime, first in FIRST[stage].items():
            for behavior, head in behaviors:
                for case in range(count):
                    _, selected = self.episode(stage, regime, first + case, 1 + case % 3, behavior, head=head)
                    records.extend(selected)
                    if case % 12 == 0:
                        print(json.dumps({'phase': stage, 'regime': regime, 'behavior': behavior, 'case': case, 'retained_rows': len(records)}), flush=True)
        x, target, mask = (self.np.stack([r[i] for r in records]) for i in range(3))
        metadata = [r[3] for r in records]
        weights = self.np.asarray(row_weights(metadata), dtype=self.np.float32)
        self.np.savez_compressed(self.out / f'{stage}-data.npz', features=x, target=target, valid=mask, weights=weights)
        for index, row in enumerate(metadata):
            self.emit(f'{stage}-rows.jsonl', {'row_index': index, **row})
        self.sync()
        return x, target, mask, metadata

    def validation(self, model, data):
        torch = self.torch
        x = data[0]
        total, matches = 0., 0
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(x), 128):
                xs, ys, ms, ws = (v[offset:offset + 128] for v in data)
                loss = self.model.training_losses(model, xs, ys, ms)
                require(bool(torch.isfinite(loss).all()), 'finite validation losses')
                total += float((loss * ws).sum())
                costs = model(xs).masked_fill(~ms, float('inf'))
                matches += int((costs.argmin(1) == ys.argmax(1)).sum())
        return {'weighted_ce': total / len(x), 'argmax_agreement': matches / len(x)}

    def fit(self, stage, training, validation):
        np, torch = self.np, self.torch
        tensors = lambda d: (torch.from_numpy(d[0]), torch.from_numpy(d[1]), torch.from_numpy(d[2]),
                             torch.from_numpy(np.asarray(row_weights(d[3]), dtype=np.float32)))
        tx, ty, tm, tw = tensors(training)
        valid = tensors(validation)
        row_hash = hashlib.sha256(json.dumps(training[3], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        weight_hash = hashlib.sha256(tw.numpy().tobytes()).hexdigest()
        data_hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in
                       (('features', tx), ('target', ty), ('valid', tm), ('weights', tw))}
        heads = {}
        for seed in SEEDS:
            for family in FAMILIES:
                fit_id = f'{family}@{seed}'
                self.context = {'phase': 'fit', 'stage': stage, 'fit_id': fit_id}
                if stage == 'initial':
                    model = self.model.make_head(KINDS[family], seed)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
                    self.models[fit_id], self.optimizers[fit_id] = model, optimizer
                model, optimizer = self.models[fit_id], self.optimizers[fit_id]
                before_steps = [int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
                curve, rng = [], np.random.default_rng(seed + (0 if stage == 'initial' else 100000))
                start = time.perf_counter()
                for epoch in range(1, 41):
                    model.train()
                    order = rng.permutation(len(tx))
                    self.emit('epoch-orders.jsonl', {'stage': stage, 'fit_id': fit_id, 'epoch': epoch,
                              'rows': len(tx), 'sha256': hashlib.sha256(order.tobytes()).hexdigest()})
                    total = 0.
                    for offset in range(0, len(order), 128):
                        idx = order[offset:offset + 128]

                        def update(idx=idx, model=model, optimizer=optimizer):
                            loss = (self.model.training_losses(model, tx[idx], ty[idx], tm[idx]) * tw[idx]).mean()
                            require(bool(torch.isfinite(loss)), 'finite weighted train loss')
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'finite present gradients')
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                            optimizer.step()
                            return float(loss.detach())
                        total += self.call('optimizer_update', update) * len(idx)
                        self.receipt['training_updates'] += 1
                    if epoch % 5 == 0:
                        value = self.validation(model, valid)
                        record = {'stage': stage, 'fit_id': fit_id, 'epoch': epoch, 'training_weighted_ce': total / len(tx), **value}
                        curve.append(record)
                        self.emit('fit-curves.jsonl', record)
                        print(json.dumps(record), flush=True)
                exported = self.model.export_head(model)
                path = self.out / f'{stage}-{family}-{seed}.npz'
                np.savez_compressed(path, **exported)
                restored = self.restore(path)
                runtime_head = self.model.FrozenHead(restored)
                model.eval()
                with torch.no_grad():
                    reference = model(valid[0][:min(16, len(valid[0]))]).numpy()
                actual = np.stack([runtime_head.scores(validation[0][i]) for i in range(len(reference))])
                errors = np.abs(actual - reference)
                require(bool(np.all(errors <= 2e-5 + 2e-5 * np.abs(reference))), 'saved NumPy/Torch cost tolerance')
                after_steps = [int(v['step'].item()) for v in optimizer.state.values() if 'step' in v]
                expected_delta = 40 * math.ceil(len(tx) / 128)
                require(len(set(after_steps)) == 1 and after_steps[0] == (before_steps[0] if before_steps else 0) + expected_delta,
                        'actual Adam state continuity and complete updates')
                fit = {'stage': stage, 'fit_id': fit_id, 'family': family, 'seed': seed, 'epochs': 40,
                       'training_rows': len(tx), 'training_episodes': len({r['episode_id'] for r in training[3]}),
                       'row_order_sha256': row_hash, 'row_weights_sha256': weight_hash,
                       'training_array_sha256': data_hashes,
                       'optimizer_steps_before': before_steps, 'optimizer_steps_after': after_steps,
                       'optimizer_continued': stage == 'final', 'checkpoint': path.name, 'checkpoint_sha256': sha(path),
                       'fit_seconds': time.perf_counter() - start, 'export_max_abs_error': float(errors.max()),
                       'export_validation_rows': list(range(len(reference))), 'export_atol': 2e-5, 'export_rtol': 2e-5,
                       'export_action_identity_asserted': False, 'curve': curve}
                self.emit('fits.jsonl', fit)
                self.receipt['completed_stage_fits'] += 1
                heads[fit_id] = runtime_head
                self.sync()
        return heads

    def restore(self, path):
        with self.np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        return {k: v.item() if v.ndim == 0 else v for k, v in values.items()}

    def evaluate(self):
        heads, setup = {}, {}
        for arm in ARMS[:-1]:
            family, seed = arm.split('@')
            path = self.out / f'final-{"dense" if family == "dense_ensemble" else family}-{seed}.npz'
            tick = time.perf_counter()
            heads[arm] = self.model.FrozenHead(self.restore(path))
            setup[arm] = {'seconds': time.perf_counter() - tick, 'checkpoint': path.name, 'sha256': sha(path), 'allocated_episodes': 72,
                          'storage': heads[arm].storage_bytes()}
        write(self.out / 'inference-setup.json', setup)
        rows, sources, uniforms = [], {}, {}
        for regime, seed, block, hit, arm in evaluation_order():
            row, _ = self.episode('eval', regime, seed, hit, arm, block=block, head=heads.get(arm), ensemble=arm.startswith('dense_ensemble@'))
            allocation = (setup[arm]['seconds'] / 72 + self.model_module_setup_seconds / 648) if arm in setup else 0.
            row['setup_allocation_seconds'] = allocation
            row['controller_seconds'] += allocation
            # Episodes already journal physical timings; allocated final rows are
            # separate so no append-only payload is rewritten after publication.
            self.emit('evaluation.jsonl', row)
            key = (regime, seed)
            source = row['source_evaluation_only']
            require(key not in sources or sources[key] == source, 'matched sampled source')
            sources[key] = source
            for draw in row['draws_evaluation_only']:
                index = (*key, draw['channel'], draw['index'])
                require(index not in uniforms or uniforms[index] == draw['uniform'], 'matched categorical uniform channel')
                uniforms[index] = draw['uniform']
            rows.append(row)
            self.receipt['completed_episodes'] = len(rows)
            if len(rows) % 10 == 0:
                print(json.dumps({'phase': 'eval', 'episodes': len(rows), 'native_steps': self.calls['native_step']['returned']}), flush=True)
        result = summary(rows, self.mixtures)
        result.update(inference_setup=setup, shared_setup_seconds=self.shared_setup_seconds,
                      model_module_setup_seconds=self.model_module_setup_seconds, model_module_allocation_episodes=648,
                      calls=self.calls, paired_source_cases=len(sources), paired_uniforms=len(uniforms),
                      all_public_beliefs_exact=True, scope='One supervised full-state readout pilot. Lambda3/4 trained; lambda5 unseen known kernel. '
                      'No new learned memory, no calibrated probabilities, no novelty claim. Every final censored/found update retained.')
        write(self.out / 'summary.json', result)
        return result

    def body(self):
        self.setup()
        self.qualify()
        self.models, self.optimizers = {}, {}
        phase_cost = {}
        tick = time.perf_counter()
        train = self.dataset('train')
        phase_cost['train_collection_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        valid = self.dataset('valid')
        phase_cost['validation_collection_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        initial = self.fit('initial', train, valid)
        phase_cost['initial_fitting_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        dagger = self.dataset('dagger', initial)
        phase_cost['dagger_collection_seconds'] = time.perf_counter() - tick
        pooled = tuple(self.np.concatenate((train[i], dagger[i])) for i in range(3)) + (train[3] + dagger[3],)
        weights = self.np.asarray(row_weights(pooled[3]), dtype=self.np.float32)
        self.np.savez_compressed(self.out / 'pooled-weights.npz', weights=weights)
        write(self.out / 'pooling.json', {'order': ['train', 'dagger'], 'rows': len(pooled[0]),
              'episodes': len({r['episode_id'] for r in pooled[3]}), 'shared_for_all_six_final_fits': True,
              'datasets': {name: descriptor(self.out / f'{name}-data.npz') for name in ('train', 'dagger')},
              'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest()})
        tick = time.perf_counter()
        self.fit('final', pooled, valid)
        phase_cost['final_fitting_seconds'] = time.perf_counter() - tick
        del train, valid, dagger, pooled, initial
        self.models.clear()
        self.optimizers.clear()
        write(self.out / 'training-costs.json', phase_cost)
        return self.evaluate()

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists(), 'exclusive absolute output')
        self.out.mkdir(parents=True, exist_ok=False)
        primary = None
        try:
            self.bind()
            result = self.body()
            require(authenticate(self.args) == self.plan and sha(self.args.supervision) == self.receipt['supervision_sha256'], 'unchanged source/input/launch closure')
            require(not self.pending and all(v['attempted'] == v['returned'] for v in self.calls.values())
                    and self.calls['native_reset']['returned'] == 1115 and self.receipt['collection_episodes'] == 384
                    and self.receipt['completed_stage_fits'] == 12 and self.receipt['completed_episodes'] == EPISODES,
                    'all fits, episodes and actual work completed')
            self.receipt.update(status='completed', pilot_continuation=result['pilot_continuation'])
            self.check()
        except BaseException as error:
            primary = error
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            finalization = []
            try:
                self.sync()
                for stream in self.handles.values():
                    stream.close()
            except BaseException as error:  # noqa: BLE001 - preserve primary error and remaining work.
                finalization.append(f'Journal finalization: {error!r}')
            try:
                end = self.clock.now_ns() if self.clock else None
                self.receipt.update(calls=self.calls, pending=self.pending, context=self.context,
                    started_ns=self.start, finished_ns=end, clock_backend=self.clock.backend if self.clock else None,
                    wall_seconds=(end - self.start) / 1e9 if end is not None and self.start is not None else None,
                    artifact_io_seconds=self.io_seconds,
                    files={p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()})
                self.check()
            except BaseException as error:  # noqa: BLE001 - a cap failure must still get a failure receipt.
                finalization.append(f'Final clock/hash/limit check: {error!r}')
                self.receipt.setdefault('started_ns', self.start)
                self.receipt.setdefault('finished_ns', None)
                self.receipt.setdefault('wall_seconds', None)
            if finalization:
                self.receipt.update(status='failed', finalization_errors=finalization,
                                    calls=self.calls, pending=self.pending, context=self.context)
            try:
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as error:  # noqa: BLE001 - retain the original execution exception.
                finalization.append(f'Receipt publication: {error!r}')
            if finalization:
                if primary is not None:
                    primary.add_note('; '.join(finalization))
                else:
                    raise RuntimeError('; '.join(finalization))
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
