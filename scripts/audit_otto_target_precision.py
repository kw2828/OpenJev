"""Independent saved-record audit of the fixed R16-versus-R64 comparison.

Admission requires an external frozen audit plan before outcome files are read.
Only saved arrays, scalar arithmetic and deterministic NumPy RNG reconstruction
are used. No teacher, sampler, model, optimizer, simulator or producer is called.

Derived from the pinned successful teacher-learning V2 auditor. It verifies
new IDs 16-63, their exact union with immutable IDs 0-15, common R16 scaling,
matched training and all 33 unchanged scientific rules on the fresh schedule.
Historical failures remain retained; no outcome determines audit allocation.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import resource
import signal
import sys
import traceback
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-target-precision-saved-audit-v1'
SELF = 'scripts/audit_otto_target_precision.py'
BASE_AUDITOR = 'scripts/audit_otto_teacher_learning_v2.py'
BASE_PIN = '41717556f04de9d616cb45923c87d42b69386770c4e90ea575918c1959da99e3'
PRESERVED = {
    'output/otto-teacher-learning-v1/audit-plan-01.json': '9a39525be6c5b755d1f9312bb15fe6bf886886535e1ee29f2a447f58dbe29682',
    'output/otto-teacher-learning-v1/audit-01/receipt.json': '4c6e8f7604bdfc0a2c99e20170359ce6393b65398fa20836ff1d1ae6c3db78e2',
    'output/otto-teacher-learning-v1/audit-supervision-01.launch.json': 'ee381f500992145cef585cc8a5d877d2ce808a99299a378ac27ad4b69d4be567',
    'output/otto-teacher-learning-v1/audit-supervision-01.terminal.json': '7b141dece0cc5ce54555dfde8746ad900cc439736ff3d160db6d84b8061f83e8',
    'output/otto-teacher-learning-v1/target-centering-diagnosis-01.json': '415ae77116eea64d620bbbfd6c0b6f61682030f9912b8b2e59fa6c3a65cc0d06',
}
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
SUPERVISOR_PIN = '610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144'
PRODUCER = 'scripts/study_otto_target_precision.py'
OLD = 'output/otto-teacher-learning-v1/run-01'
PLAN_PIN = '921318b9c978b208989ad66210a06adad8f104c827351eda1850f080d9f8a46a'
COHORT = 'output/otto-teacher-cohort-v1/run-01/anchors.npz'
SYMM = 'output/otto-symmetry-head-v1/run-01'
LIMITS = {'seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 64 * 1024**2}
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
H, LABEL_SEED, SEEDS = 2188, 19000002, (20101, 20102, 20103)
KINDS = ('r16', 'r64')
ARMS = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in KINDS) + ('analytic_inbounds',)
FIRST = {'lambda3': 1050001, 'lambda4': 1060001, 'lambda5': 1070001}
ROOT_FILES = {'started.json', 'panels.jsonl', 'combined-labels.jsonl', 'collection.json', 'training-data.npz', 'training-data.json',
              'training.jsonl', 'fits.jsonl', 'training-summary.json', 'deployment.json', 'evaluation.jsonl',
              'eval-transitions.jsonl.gz', 'work.jsonl', 'summary.json'}
PAYLOADS = ROOT_FILES | {f'panel-{i:03d}.jsonl.gz' for i in range(558)} | {
    f'{kind}-{seed}-{phase}.npz' for kind in KINDS for seed in SEEDS for phase in ('initial', 'final')}
SHAPES = {'weight0': (32, 2836), 'bias0': (32,), 'weight1': (16, 32), 'bias1': (16,),
          'weight2': (4, 16), 'bias2': (4,)}
LIMITATIONS = [
    'Saved-record audit only: zero fresh teacher, sampler, model, optimizer or simulator calls.',
    'Teacher and deployed choices are checked against saved scores; scores are not regenerated.',
    'Feature-generation and optimizer execution are inherited from authenticated source/process evidence.',
    'Checkpoint arrays and recorded TRAIN parity are checked; neural predictions and gradients are not rerun.',
    'Continuation source/hit streams and geometry are reconstructed, but public posterior filtering is not rerun.',
    'Evaluation posterior witnesses are linked, with terminal point masses checked; intermediate beliefs are not regenerated.',
    'Timing, actual native execution and fsync completion remain authenticated original-process evidence.',
    'Historical audits are authenticated as inputs, not rerun; no new efficacy or architecture experiment is admitted.',
    'Original R16 event-stream replay is inherited from its completed V2 audit; original saved costs are rejoined and reduced.',
]


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f'duplicate JSON key {key}')
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError(f'nonfinite JSON constant {value}')


def decode(value):
    return json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)


def regular(path):
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink path required')
    return path


def relative(name):
    value = Path(name)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('contained manifest path required')
    return regular(ROOT / value)


def digest(path, check=lambda: None):
    regular(path)
    hashed, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            hashed.update(block)
            size += len(block)
    return {'sha256': hashed.hexdigest(), 'bytes': size}


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def schedule():
    for regime_index, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            shift = (regime_index * 24 + case) % 7
            for arm in ARMS[shift:] + ARMS[:shift]:
                yield regime, first + case, case, 1 + case % 3, case // 3, arm


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.deadline = None
        self.context, self.bound = {}, {}
        self.counts = collections.Counter()
        self.receipt = {'version': VERSION, 'status': 'started', 'agreement': False,
                        'limitations': LIMITATIONS, 'failures': [], 'teacher_calls': 0,
                        'sampler_calls': 0, 'model_calls': 0, 'optimizer_calls': 0, 'native_calls': 0,
                        'requires_successful_original_supervisor': True}

    def require(self, value, message):
        self.counts['checks'] += 1
        if not value:
            raise ValueError(message)

    def equal(self, left, right, message):
        self.require(encoded(left) == encoded(right), message)

    def near(self, left, right, message, *, rtol=1e-11, atol=1e-12):
        self.require(type(left) in (int, float) and math.isfinite(left)
                     and math.isclose(left, right, rel_tol=rtol, abs_tol=atol), message)

    def finite(self, value, message, *, nonnegative=True):
        self.require(type(value) in (int, float) and math.isfinite(value)
                     and (value >= 0 or not nonnegative), message)

    def check(self):
        self.require(self.clock.now_ns() < self.deadline, 'audit original suspend-inclusive deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        self.require(rss <= LIMITS['rss_bytes'], 'audit RSS bound')
        self.require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                     <= LIMITS['output_bytes'], 'audit output bound')

    def bind(self, path, expected=None):
        actual = digest(path, self.check)
        if expected is not None:
            self.equal(actual, expected, f'file descriptor {path}')
        self.bound[str(path)] = actual
        return actual

    def read(self, path):
        self.check()
        self.require(path.stat().st_size <= 32 * 1024**2, 'bounded JSON document')
        return decode(path.read_bytes())

    def lines(self, path, *, maximum, count_limit):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            count = 0
            while raw := stream.readline(maximum + 1):
                count += 1
                self.require(len(raw) <= maximum and raw.endswith(b'\n') and count <= count_limit,
                             'bounded complete JSONL record')
                if count % 1024 == 1:
                    self.check()
                yield decode(raw)

    def exhaust(self, iterator, message):
        end = object()
        self.require(next(iterator, end) is end, message)

    def admission(self):
        self.require(digest(self.args.audit_plan)['sha256'] == self.args.audit_plan_sha256, 'external audit-plan pin')
        plan = decode(self.args.audit_plan.read_bytes())
        self.require(plan['version'] == VERSION and plan['status'] == 'frozen_before_saved_readback'
                     and plan['limits'] == LIMITS and plan['environment'] == THREADS, 'frozen audit scope/budget')
        self.require(plan['sources'] == {SELF: digest(relative(SELF))['sha256'], CLOCK: CLOCK_PIN,
                                        SUPERVISOR: SUPERVISOR_PIN, BASE_AUDITOR: BASE_PIN}
                     and digest(relative(CLOCK))['sha256'] == CLOCK_PIN
                     and digest(relative(SUPERVISOR))['sha256'] == SUPERVISOR_PIN
                     and digest(relative(BASE_AUDITOR))['sha256'] == BASE_PIN,
                     'auditor and qualified clock/supervisor pins')
        self.equal(plan['preserved_inputs'], PRESERVED, 'fixed failed audit and diagnosis lineage')
        self.require(set(plan['inputs']) == {'plan', 'worker', 'terminal'}, 'exact audit input roles')
        self.require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'one audit numerical thread')
        spec = importlib.util.spec_from_file_location('_teacher_learning_saved_clock', relative(CLOCK))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        self.deadline = self.start + LIMITS['seconds'] * 10**9
        self.require(self.clock.backend in ('mach_continuous_time', 'CLOCK_BOOTTIME'), 'native suspend-inclusive clock')
        launch = self.read(self.args.supervision)
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(launch['version'] == 'dialogue-observation-supervision-v2'
                     and command == [sys.executable, *sys.argv]
                     and launch['pid'] == os.getpid() and launch['pgid'] == os.getpgrp()
                     and launch['parent_pid'] == os.getppid()
                     and launch['cwd'] == str(ROOT) == str(Path.cwd())
                     and type(launch['cap_seconds']) is int and launch['cap_seconds'] == LIMITS['seconds']
                     and launch['clock_backend'] == self.clock.backend
                     and launch['clock_source_sha256'] == CLOCK_PIN
                     and launch['watchdog_sha256'] == SUPERVISOR_PIN
                     and launch['deadline_ns'] == launch['started_ns'] + LIMITS['seconds'] * 10**9
                     and launch['started_ns'] <= self.start < launch['deadline_ns'],
                     'original audit supervision identity and deadline')
        self.deadline = launch['deadline_ns']
        self.receipt.update(supervision_sha256=self.bind(self.args.supervision)['sha256'],
                            original_deadline_ns=self.deadline)
        self.audit_plan = plan
        self.self_source = self.bind(relative(SELF))
        self.bind(self.args.audit_plan)
        self.bind(relative(CLOCK))
        self.bind(relative(SUPERVISOR))
        self.bind(relative(BASE_AUDITOR))
        for name, pin in PRESERVED.items():
            self.require(self.bind(relative(name))['sha256'] == pin, 'unchanged failed audit evidence')
        self.receipt.update(base_auditor_sha256=BASE_PIN, preserved_inputs=PRESERVED,
                            scope='New R48 saved event replay, exact R16/R64 union, fixed common scale and complete matched study.')
        write(self.out / 'started.json', {'audit_plan_sha256': self.args.audit_plan_sha256,
              'source': self.self_source, 'limits': LIMITS, 'inputs': plan['inputs'],
              'base_auditor_sha256': BASE_PIN, 'preserved_inputs': PRESERVED,
              'started_ns': self.start, 'clock_backend': self.clock.backend,
              'command': [sys.executable, *sys.argv], 'launch': launch,
              'request': {k: str(v) for k, v in vars(self.args).items()}})

    def authenticate(self):
        entries = self.audit_plan['inputs']
        for role, value in entries.items():
            actual = self.bind(regular(Path(value['path'])))
            self.require(actual['sha256'] == value['sha256'], f'external {role} pin')
        self.require(entries['plan']['sha256'] == PLAN_PIN, 'fixed empirical plan')
        self.plan = self.read(Path(entries['plan']['path']))
        self.worker = self.read(Path(entries['worker']['path']))
        terminal = self.read(Path(entries['terminal']['path']))
        self.run = Path(entries['worker']['path']).parent
        plan, worker = self.plan, self.worker
        self.receipt['producer_status'] = worker.get('status')
        self.require(plan['status'] == 'frozen_before_label_generation' and worker['status'] == 'completed',
                     'completed producer required; failed-study review needs separate admission')
        self.require(plan['version'] == worker['version'] == 'otto-target-precision-v1'
                     and worker['plan_sha256'] == PLAN_PIN and worker['requires_successful_original_supervisor'] is True,
                     'producer identity')
        self.require(worker['pending'] == [] and worker['pending_panel'] is None and worker['pending_episode'] is None,
                     'producer work closure')
        self.equal([worker[k] for k in ('completed_panels', 'completed_fits', 'completed_episodes')],
                   [558, 6, 504], 'complete scientific stages')
        self.equal(plan['limits'], {'native_seconds': 10800, 'rss_bytes': 4 * 1024**3,
                                   'output_bytes': 32 * 1024**3}, 'producer budgets')
        self.require(set(worker['files']) == PAYLOADS
                     and {p.name for p in self.run.iterdir()} == PAYLOADS | {'receipt.json'}, 'exact 584 payload closure')
        for name, desc in worker['files'].items():
            self.bind(self.run / name, desc)
        for name, pin in plan['sources'].items():
            self.require(self.bind(relative(name))['sha256'] == pin, 'frozen producer source')
        for name, desc in plan['inputs'].items():
            self.bind(relative(name), desc)
        self.require(len(plan['sources']) == 158 and len(plan['inputs']) == 1144, 'complete precision-study manifest')
        self.counts.update(payloads=584, sources=len(plan['sources']), inputs=len(plan['inputs']))
        started = self.read(self.run / 'started.json')
        launch, request = started['launch'], started['request']
        launch_path = regular(Path(request['supervision']))
        self.require(self.bind(launch_path)['sha256'] == worker['supervision_sha256'], 'original launch pin')
        self.equal(self.read(launch_path), launch, 'embedded launch')
        self.require(all(terminal[k] == v for k, v in launch.items()), 'original terminal/launch identity')
        self.require(terminal['status'] == 'completed' and terminal['returncode'] == 0
                     and terminal['timed_out'] is False and terminal['error'] is None
                     and terminal['clock_error'] is None and terminal['timing_available'] is True
                     and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
                     and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True,
                     'successful original parent and cleanup')
        self.require(request['plan'] == entries['plan']['path'] and request['plan_sha256'] == PLAN_PIN
                     and request['output'] == str(self.run), 'request identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(command[:3] == [plan['python_executable'], str(relative(PRODUCER)), 'run'], 'original process command')
        self.require(len(command[3:]) == 8 and len(set(command[3::2])) == 4, 'exact original command flags')
        self.equal(dict(zip(command[3::2], command[4::2], strict=True)),
                   {'--plan': entries['plan']['path'], '--plan-sha256': PLAN_PIN,
                    '--supervision': str(launch_path), '--output': str(self.run)}, 'original bound paths')
        self.require(launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
                     and launch['cap_seconds'] == 10800 and launch['clock_source_sha256'] == CLOCK_PIN
                     and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                     and launch['clock_backend'] == worker['clock_backend']
                     and launch['deadline_ns'] == launch['started_ns'] + 10800 * 10**9,
                     'original process and deadline')
        self.require(launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns']
                     <= terminal['finished_ns'] < launch['deadline_ns'], 'worker/parent time closure')
        self.near(worker['wall_seconds'], (worker['finished_ns'] - worker['started_ns']) / 1e9, 'worker time arithmetic')
        self.equal(terminal['elapsed_ns'], terminal['finished_ns'] - terminal['started_ns'], 'parent elapsed nanoseconds')
        self.near(terminal['wall_seconds'], terminal['elapsed_ns'] / 1e9, 'parent seconds')
        self.require(worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                     and sum(p.stat().st_size for p in self.run.iterdir()) <= plan['limits']['output_bytes'], 'producer resource closure')
        self.require(sys.executable == plan['python_executable'] and platform.python_version() == plan['python_version'],
                     'qualified audit interpreter')
        self.rows = plan['selections']
        self.require(len(self.rows) == 558 and [r['anchor_id'] for r in self.rows] == list(range(558))
                     and len({r['episode_id'] for r in self.rows}) == 144, 'complete ordered public cohort')
        historical_worker = self.read(relative(f'{OLD}/receipt.json'))
        historical_audit = self.read(relative('output/otto-teacher-learning-v1/audit-02/receipt.json'))
        self.require(historical_worker['status'] == historical_audit['status'] == 'completed'
                     and historical_audit['agreement'] is True
                     and historical_audit['version'] == 'otto-teacher-learning-saved-audit-v2',
                     'original R16 provenance inherits its completed independent saved audit')
        self.equal(historical_audit['source'], self.bound[str(relative(BASE_AUDITOR))], 'original completed auditor source')
        self.require(historical_worker['completed_panels'] == 558 and historical_worker['completed_fits'] == 6
                     and historical_worker['completed_episodes'] == 504, 'complete original study is preserved')
        self.receipt.update(producer_inputs=entries, producer_parent_seconds=terminal['wall_seconds'])

    def arrays(self):
        import numpy as np
        self.np = np
        self.require(np.__version__ == self.plan['all_distributions']['numpy'], 'qualified saved-array version')
        with np.load(relative(COHORT), allow_pickle=False) as saved:
            self.require(set(saved.files) == {'beliefs', 'kernel_lambda3', 'kernel_lambda4'}, 'cohort array schema')
            self.beliefs = saved['beliefs']
        self.require(self.beliefs.shape == (558, 53, 53) and self.beliefs.dtype == np.float64
                     and np.isfinite(self.beliefs).all() and (self.beliefs >= 0).all(), 'finite captured public beliefs')
        self.kernels, self.mixtures = {}, {}
        for regime in FIRST:
            with np.load(relative(f'{SYMM}/kernel-{regime}.npz'), allow_pickle=False) as saved:
                self.require(set(saved.files) == {'likelihood', 'initial_hit_weights'}, 'public kernel schema')
                kernel, mixture = saved['likelihood'], saved['initial_hit_weights']
            self.require(kernel.dtype == np.float64 and kernel.shape == (4, 107, 107)
                         and np.isfinite(kernel).all() and ((kernel >= 0) & (kernel <= 1)).all(), 'public kernel values')
            self.require(mixture.shape == (4,) and mixture[0] == 0 and (mixture[1:] > 0).all(), 'positive hit mixture')
            self.near(float(mixture.sum()), 1, 'hit mixture normalization')
            self.kernels[regime], self.mixtures[regime] = kernel, mixture.tolist()
        for row, belief in zip(self.rows, self.beliefs, strict=True):
            self.equal(row['posterior'], {'sha256': hashlib.sha256(belief.tobytes()).hexdigest(),
                                         'mass': float(belief.sum())}, 'selected posterior witness')
            self.packet(row['public'], row['prefix_index'])
        with np.load(self.run / 'training-data.npz', allow_pickle=False) as saved:
            self.data = {key: saved[key] for key in saved.files}
        fields = ('centered_float64', 'scaled_float32', 'weights_float64', 'weights_float32', 'allowed')
        expected = {'features', 'r16_costs', 'r64_costs', 'allowed', 'anchor_ids'} | {
            f'{kind}_{key}' for kind in KINDS for key in fields}
        self.require(set(self.data) == expected, 'exact TRAIN array schema')
        features = self.data['features']
        self.require(features.dtype == np.float32 and features.shape == (558, 2836)
                     and np.isfinite(features).all(), 'finite common features')
        self.require(self.data['anchor_ids'].dtype == np.int64
                     and np.array_equal(self.data['anchor_ids'], np.arange(558)), 'TRAIN anchor alignment')
        self.mask = np.array([[a in row['public']['valid_actions'] for a in range(4)] for row in self.rows])
        self.require(self.data['allowed'].dtype == np.bool_ and np.array_equal(self.data['allowed'], self.mask),
                     'public geometric target mask')
        self.data_info = self.read(self.run / 'training-data.json')
        self.equal(self.data_info['rows'], self.rows, 'TRAIN row provenance')
        self.equal(self.data_info['file'], {'path': 'training-data.npz', **self.worker['files']['training-data.npz']},
                   'TRAIN array descriptor')
        self.require(self.data_info['episodes'] == 144 and self.data_info['features_sha256']
                     == hashlib.sha256(features.tobytes()).hexdigest(), 'TRAIN features hash')
        cache_name = f'{OLD}/training-data.npz'
        self.equal(self.data_info['original_training_cache'], {'path': cache_name, **self.plan['inputs'][cache_name]},
                   'original R16 cache descriptor')
        with np.load(relative(cache_name), allow_pickle=False) as saved:
            inherited = {'features', 'analytic_costs', 'continuation_means', 'allowed', 'anchor_ids'} | {
                f'{kind}_{key}' for kind in ('analytic', 'continuation') for key in fields}
            self.require(set(saved.files) == inherited, 'inherited original cache schema')
            self.reference_costs = saved['continuation_means']
            self.reference = {key: saved[f'continuation_{key}'] for key in fields}
            for key in ('features', 'allowed', 'anchor_ids'):
                self.require(self.data[key].dtype == saved[key].dtype and self.data[key].shape == saved[key].shape
                             and self.data[key].tobytes() == saved[key].tobytes(), 'unchanged original features/mask/order')
        original_info = self.read(relative(f'{OLD}/training-data.json'))
        self.equal(original_info['rows'], self.rows, 'unchanged original row selection')
        self.reference_scale = original_info['scales']['continuation']
        self.finite(self.reference_scale, 'original R16 global scale')
        self.require(self.reference_scale >= 1e-8, 'original scale floor')
        self.historical_collection = self.read(relative(f'{OLD}/collection.json'))

    def packet(self, value, step):
        self.require(set(value) == {'position', 'hit', 'done', 'step', 'valid_actions'}, 'public packet schema')
        position = value['position']
        self.require(isinstance(position, list) and len(position) == 2
                     and all(type(v) is int and 0 <= v < 53 for v in position), 'public position')
        self.require(type(value['step']) is int and value['step'] == step and type(value['done']) is bool
                     and type(value['hit']) is int and (value['hit'] == -2 if value['done'] else 0 <= value['hit'] < 4),
                     'public step and hit sentinel')
        allowed = [] if value['done'] else [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]
        self.equal(value['valid_actions'], allowed, 'public eligibility')
        return value

    def scalar_draw(self, probabilities, uniform):
        cumulative, total = [], 0.0
        for value in probabilities:
            self.finite(float(value), 'categorical probability')
            total += float(value)
            cumulative.append(total)
        self.require(abs(total - 1) <= 1e-10 and 0 <= uniform < 1, 'categorical domain')
        chosen = bisect.bisect_right([value / total for value in cumulative], uniform)
        self.require(chosen < len(cumulative) and probabilities[chosen] > 0, 'positive selected support')
        return chosen, total

    def panels(self):
        np = self.np
        boundaries = iter(self.lines(self.run / 'panels.jsonl', maximum=128 * 1024, count_limit=1116))
        old_boundaries = iter(self.lines(relative(f'{OLD}/panels.jsonl'), maximum=128 * 1024, count_limit=1116))
        combinations = iter(self.lines(self.run / 'combined-labels.jsonl', maximum=256 * 1024, count_limit=558))
        self.label_means = np.full((558, 4), np.inf, dtype=np.float64)
        calls, totals = collections.Counter(), collections.Counter()
        for row, belief in zip(self.rows, self.beliefs, strict=True):
            anchor = row['anchor_id']
            self.context = {'phase': 'panels', 'anchor_id': anchor}
            self.equal(next(boundaries), {'event': 'attempt', 'phase': 'sampling', 'anchor_id': anchor}, 'durable panel attempt')
            panel = next(boundaries)
            self.require(panel['event'] == 'return' and panel['phase'] == 'sampling'
                         and panel['anchor_id'] == anchor, 'durable panel return')
            filename = f'panel-{anchor:03d}.jsonl.gz'
            self.equal(panel['file'], {'path': filename, **self.worker['files'][filename]}, 'closed panel file')
            self.finite(panel['seconds'], 'panel measured interval')
            actions = row['public']['valid_actions']
            self.equal(next(old_boundaries), {'event': 'attempt', 'phase': 'sampling', 'anchor_id': anchor},
                       'immutable original panel attempt')
            old_panel = next(old_boundaries)
            self.require(old_panel['event'] == 'return' and old_panel['phase'] == 'sampling'
                         and old_panel['anchor_id'] == anchor, 'immutable original panel return')
            old_name = f'{OLD}/{filename}'
            self.equal(old_panel['file'], {'path': filename, **self.plan['inputs'][old_name]}, 'original R16 panel byte join')
            self.checked_records(old_panel['records'], actions, range(16))
            records = panel['records']
            self.equal([(r['replicate_id'], r['first_action']) for r in records],
                       list(itertools.product(range(16, 64), actions)), 'complete canonical new R48 panel')
            by_key = {(r['replicate_id'], r['first_action']): r for r in records}
            for record in records:
                self.require(set(record) == {'replicate_id', 'first_action', 'steps', 'found'}
                             and type(record['steps']) is int and 1 <= record['steps'] <= H
                             and type(record['found']) is bool and (record['found'] or record['steps'] == H),
                             'honest capped continuation')
            events = iter(self.lines(self.run / filename, maximum=512, count_limit=1696844106))
            base = {'version': 'otto-teacher-rollouts-v1', 'seed': LABEL_SEED, 'anchor_id': anchor}
            operation_id, event_count = 0, 0

            def event(name, fields=None, events=events, base=base):
                nonlocal event_count
                value = next(events)
                event_count += 1
                self.require(all(value.get(k) == v for k, v in {**base, 'event': name, **(fields or {})}.items()),
                             'sampler event identity/order')
                return value

            def operation(name, context=None, base=base, event=event):
                nonlocal operation_id
                operation_id += 1
                identity = {'operation_id': operation_id, 'operation': name, **(context or {})}
                self.equal(event('attempt', identity), {**base, 'event': 'attempt', **identity}, 'exact operation attempt')
                result = event('return', identity)
                calls[name] += 1
                return result

            self.equal(operation('anchor_snapshot')['public'], row['public'], 'validation snapshot packet')
            for replicate in range(16, 64):
                context = {'replicate_id': replicate}
                operation('source_generator', context)
                source_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F54544F, LABEL_SEED, anchor, replicate, 0])))
                source_draw = operation('source_draw', context)
                uniform = float(source_rng.random())
                index, mass = self.scalar_draw(belief.reshape(-1), uniform)
                self.equal([source_draw['uniform'], source_draw['selected_index'], source_draw['cdf_mass'], source_draw['source']],
                           [uniform, index, mass, list(divmod(index, 53))], 'source RNG/CDF/source witness')
                source = source_draw['source']
                for first in actions:
                    context = {'replicate_id': replicate, 'first_action': first}
                    self.equal(operation('teacher_snapshot', context)['public'], row['public'], 'fresh continuation snapshot')
                    operation('hit_generator', context)
                    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F54544F, LABEL_SEED, anchor, replicate, 1])))
                    record, public = by_key[replicate, first], row['public']
                    for step in range(1, record['steps'] + 1):
                        step_context = {**context, 'local_step': step, 'from_step': public['step']}
                        if step == 1:
                            forced = event('forced_action', step_context)
                            self.require(forced['action'] == first, 'counted forced first action')
                            action = first
                        else:
                            choice = operation('teacher_choose', step_context)
                            action = self.saved_choice(choice['scores'], public['valid_actions'])
                            self.require(choice['action'] == action, 'teacher selection from saved scores')
                        position = list(public['position'])
                        position[action // 2] += 2 * (action % 2) - 1
                        found = position == source
                        moved = operation('movement', step_context)
                        self.equal([moved['position'], moved['found']], [position, found], 'source-conditioned move')
                        hit = -2
                        if not found:
                            draw = operation('hit_draw', step_context)
                            probability = self.kernels[row['regime']][:, 53 + source[0] - position[0], 53 + source[1] - position[1]]
                            uniform = float(rng.random())
                            hit, mass = self.scalar_draw(probability, uniform)
                            self.equal([draw['probabilities'], draw['uniform'], draw['selected_index'], draw['cdf_mass'], draw['draw_index']],
                                       [probability.tolist(), uniform, hit, mass, step - 1], 'paired hit RNG/kernel/CDF')
                        public = {'position': position, 'hit': hit, 'done': found, 'step': public['step'] + 1,
                                  'valid_actions': [] if found else [a for a in range(4) if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}
                        self.equal(operation('teacher_update', step_context)['public'], public, 'every final packet updated')
                        self.require(not found or step == record['steps'], 'first discovery stops continuation')
                    self.require(public['done'] == record['found'], 'record outcome matches last packet')
                    self.equal(event('record', context), {**base, 'event': 'record', **record}, 'saved continuation record')
            self.equal(event('panel_complete'), {**base, 'event': 'panel_complete', 'record_count': len(records),
                                                 'operation_count': operation_id}, 'panel completion witness')
            self.exhaust(events, 'no surplus sampler events')
            steps, found = sum(r['steps'] for r in records), sum(r['found'] for r in records)
            self.require(event_count == 195 + 4 * len(records) + 8 * steps - 2 * found, 'new R48 panel event identity')
            self.reduction(panel['summary'], records, actions, range(16, 64))
            self.combined_labels(next(combinations), old_panel, panel, actions, anchor)
            totals.update(records=len(records), steps=steps, found=found, events=event_count)
        self.exhaust(boundaries, 'exact 558 panel boundaries')
        self.exhaust(old_boundaries, 'exact immutable original panel boundaries')
        self.exhaust(combinations, 'exact 558 combined panels')
        self.require(totals['records'] == 96912, 'all new continuation costs')
        expected = {'anchor_snapshot': 558, 'source_generator': 26784, 'source_draw': 26784,
                    'teacher_snapshot': 96912, 'hit_generator': 96912, 'teacher_choose': totals['steps'] - 96912,
                    'movement': totals['steps'], 'hit_draw': totals['steps'] - totals['found'], 'teacher_update': totals['steps']}
        self.equal(dict(calls), expected, 'complete sampler work counts')
        acknowledged = {key: {'attempted': value, 'returned': value} for key, value in expected.items()}
        self.equal(self.worker['sampler_calls'], acknowledged, 'worker sampler ledger')
        self.collection = self.read(self.run / 'collection.json')
        for key, value in {'panels': 558, 'records': 96912, 'steps': totals['steps'], 'found': totals['found'],
                           'censored': 96912 - totals['found'], 'sampler_events': totals['events'],
                           'reused_records': 32304, 'combined_records': 129216,
                           'replicate_ids': list(range(16, 64)), 'reducer_calls': 3348}.items():
            self.equal(self.collection[key], value, 'collection aggregate')
        self.equal(self.collection['historical_r16_collection_seconds'], self.historical_collection['seconds'],
                   'historical label cost is reused, not free')
        self.equal(self.collection['calls'], acknowledged, 'collection calls')
        self.equal(self.worker['sampler_events'], totals['events'], 'worker event count')
        self.counts.update(panels=558, continuations=96912, reused_continuations=32304,
                           combined_continuations=129216, sampler_events=totals['events'], continuation_moves=totals['steps'])

    def checked_records(self, records, actions, replicates):
        self.equal([(r['replicate_id'], r['first_action']) for r in records],
                   list(itertools.product(replicates, actions)), 'complete canonical saved cost panel')
        for record in records:
            self.require(set(record) == {'replicate_id', 'first_action', 'steps', 'found'}
                         and type(record['steps']) is int and 1 <= record['steps'] <= H
                         and type(record['found']) is bool and (record['found'] or record['steps'] == H),
                         'complete original or combined capped continuation')

    def combined_labels(self, combined, old_panel, new_panel, actions, anchor):
        records = old_panel['records'] + new_panel['records']
        self.checked_records(combined['records'], actions, range(64))
        self.equal(combined['records'], records, 'append-only exact R16 plus R48 records')
        sums = {str(a): sum(r['steps'] for r in records if r['first_action'] == a) for a in actions}
        means = [sums[str(a)] / 64 if a in actions else None for a in range(4)]
        self.equal({k: combined[k] for k in ('anchor_id', 'integer_sums', 'means', 'reused_records', 'new_records', 'total_records')},
                   {'anchor_id': anchor, 'integer_sums': sums, 'means': means,
                    'reused_records': 16 * len(actions), 'new_records': 48 * len(actions),
                    'total_records': 64 * len(actions)}, 'exact combined integer totals and denominators')
        old_name = f'{OLD}/panel-{anchor:03d}.jsonl.gz'
        self.equal(combined['source_r16'], {'path': old_name, **self.plan['inputs'][old_name]}, 'combined old source pin')
        self.equal(combined['source_r48'], new_panel['file'], 'combined new source pin')
        for action in actions:
            self.label_means[anchor, action] = means[action]
            original_mean = sum(r['steps'] for r in old_panel['records'] if r['first_action'] == action) / 16
            self.equal(float(self.reference_costs[anchor, action]), original_mean, 'original mean from preserved integer records')
        self.reduction(combined['summary_r64'], records, actions, range(64))
        self.require(len(combined['disjoint16']) == 4, 'four disjoint sixteen-replicate diagnostics')
        self.equal(old_panel['summary'], combined['disjoint16'][0]['summary'], 'original reduction retained as first block')
        selected = []
        for block, start in zip(combined['disjoint16'], (0, 16, 32, 48), strict=True):
            replicas = range(start, start + 16)
            self.equal(block['replicate_ids'], list(replicas), 'fixed disjoint block identities')
            subset = [r for r in records if r['replicate_id'] in replicas]
            self.reduction(block['summary'], subset, actions, replicas)
            totals = {a: sum(r['steps'] for r in subset if r['first_action'] == a) for a in actions}
            selected.append(min(actions, key=lambda a, totals=totals: (totals[a], a)))
        self.equal(combined['block16_actions'], selected, 'fixed numeric tie rule in each disjoint block')
        self.equal(combined['block16_action_disagreements'],
                   sum(selected[a] != selected[b] for a, b in itertools.combinations(range(4), 2)),
                   'descriptive block disagreements without filtering')

    def reduction(self, summary, records, actions, replicates):
        values = {(r['replicate_id'], r['first_action']): r for r in records}
        n, k = len(replicates), len(actions)
        totals = {a: sum(values[r, a]['steps'] for r in replicates) for a in actions}
        grand = sum(totals.values())
        self.equal([summary[key] for key in ('version', 'horizon', 'replicate_count', 'action_count', 'pair_count', 'familywise_alpha')],
                   ['otto-teacher-cost-summary-v1', H, n, k, k * (k - 1) // 2, .05], 'reduction allocation')
        self.equal([r['action'] for r in summary['actions']], actions, 'reduction action coverage')
        for row in summary['actions']:
            action = row['action']
            successes = sum(values[r, action]['found'] for r in replicates)
            mean = totals[action] / n
            for key, expected in {'mean_cost': mean, 'centered_mean_cost': (k * totals[action] - grand) / (n * k),
                                  'success_fraction': successes / n, 'censored_fraction': (n - successes) / n}.items():
                self.near(row[key], expected, 'independent action reduction')
        pair_ids = list(itertools.combinations(actions, 2))
        self.equal([(r['first_action'], r['second_action']) for r in summary['pairs']], pair_ids, 'all paired comparisons')
        radius = 2 * (H - 1) * math.sqrt(math.log(2 * len(pair_ids) / .05) / (2 * n)) if pair_ids else 0
        for row, (first, second) in zip(summary['pairs'], pair_ids, strict=True):
            delta = [values[r, first]['steps'] - values[r, second]['steps'] for r in replicates]
            total = sum(delta)
            mean, se = total / n, math.sqrt((n * sum(v * v for v in delta) - total * total) / (n * n * (n - 1)))
            lower, upper = max(1 - H, mean - radius), min(H - 1, mean + radius)
            for key, expected in {'mean_difference': mean, 'paired_standard_error': se, 'hoeffding_radius': radius}.items():
                self.near(row[key], expected, 'independent paired reduction')
            for actual, expected in zip(row['hoeffding_interval'], (lower, upper), strict=True):
                self.near(actual, expected, 'simultaneous interval')
            self.equal(row['resolved_direction'], 'first_lower' if upper < 0 else 'second_lower' if lower > 0 else 'unresolved',
                       'interval direction')

    def saved_choice(self, scores, allowed):
        self.require(len(scores) == 4 and allowed, 'four-action decision')
        for action in allowed:
            self.finite(scores[action], 'eligible saved score', nonnegative=False)
        minimum = min(scores[a] for a in allowed)
        return next(a for a in allowed if abs(float(scores[a]) - minimum) < 1e-10)

    def target_roundoff_checks(self, values, saved, mean, reference_mean, denominator):
        """Verify centered gaps without amplifying a summation-order difference.

        All comparisons use exact rationals represented by saved binary64s.
        gamma_4 bounds the four-slot sum and division mean; 4*eta covers
        subnormal rounding. Each action bound covers only rounded subtraction
        and division. Pairwise differences cancel the shared mean offset.
        The fsum comparison explicitly subtracts that offset in original units.
        """
        q = Fraction.from_float
        unit, eta = Fraction(1, 2**53), Fraction(1, 2**1074)
        gamma = 4 * unit / (1 - 4 * unit)
        exact = [q(float(value)) for value in values]
        exact_mean = sum(exact, Fraction()) / len(values)
        actual_mean, reference, divisor = q(float(mean)), q(float(reference_mean)), q(float(denominator))
        self.require(abs(actual_mean - exact_mean)
                     <= gamma * sum(map(abs, exact), Fraction()) / len(values) + 4 * eta,
                     'qualified mean within binary64 sum/division roundoff')
        bounds, outputs = [], []
        for value, target, original in zip(values, saved, exact, strict=True):
            numerator = float(value) - float(mean)
            reference_numerator = float(value) - float(reference_mean)
            output = q(float(target))
            bound = q(math.ulp(numerator)) / 2 + divisor * q(math.ulp(float(target))) / 2
            self.require(abs(output * divisor - (original - exact_mean) - (exact_mean - actual_mean)) <= bound,
                         'original-unit centering error after shared mean offset')
            self.require(abs(output * divisor - q(reference_numerator) - (reference - actual_mean))
                         <= bound + q(math.ulp(reference_numerator)) / 2,
                         'fsum reference in original units with explicit common offset')
            bounds.append(bound)
            outputs.append(output)
        for a, b in itertools.combinations(range(len(values)), 2):
            self.require(abs((outputs[a] - outputs[b]) * divisor - (exact[a] - exact[b])) <= bounds[a] + bounds[b],
                         'independent pairwise cost gaps within operation roundoff')
        self.counts['target_roundoff_rows'] += 1

    def targets(self):
        np = self.np
        identities = collections.Counter(row['episode_id'] for row in self.rows)
        weights = np.array([558 / (144 * identities[row['episode_id']]) for row in self.rows], dtype=np.float64)
        self.target_hashes = {}
        for kind, costs in (('r16', self.reference_costs), ('r64', self.label_means)):
            self.context = {'phase': 'targets', 'kind': kind}
            saved = self.data[f'{kind}_costs']
            self.require(costs.dtype == np.float64 and costs.shape == (558, 4)
                         and np.isfinite(costs[self.mask]).all() and np.isposinf(costs[~self.mask]).all()
                         and saved.dtype == np.float64 and saved.shape == costs.shape
                         and saved.tobytes() == costs.tobytes(), 'exact raw preserved or combined continuation costs')
            finite_costs = np.where(self.mask, costs, 0.0)
            counts = self.mask.sum(axis=1, dtype=np.int64)
            qualified_means = np.sum(finite_costs, axis=1, dtype=np.float64) / counts
            centered = np.where(self.mask, finite_costs - qualified_means[:, None], 0.0)
            original = self.data[f'{kind}_centered_float64']
            cast = self.data[f'{kind}_scaled_float32']
            self.require(original.dtype == np.float64 and original.shape == (558, 4)
                         and np.isfinite(original).all() and (original[~self.mask] == 0).all()
                         and original.tobytes() == centered.tobytes(),
                         'byte-exact continuation centering without per-panel range normalization')
            row_moments = []
            for i, row in enumerate(self.rows):
                actions = row['public']['valid_actions']
                values = [float(costs[i, a]) for a in actions]
                reference_mean = math.fsum(values) / len(values)
                self.target_roundoff_checks(values, [original[i, a] for a in actions], qualified_means[i], reference_mean, 1.0)
                row_moments.append(math.fsum((value - reference_mean) ** 2 for value in values) / len(values))
            independent_rms = math.sqrt(math.fsum(float(w) * v for w, v in zip(weights, row_moments, strict=True)) / 558)
            self.require(float(self.data_info['scales'][kind]).hex() == float(self.reference_scale).hex(),
                         'both arms use exactly the historical R16 scale')
            if kind == 'r16':
                squares = np.sum(centered * centered, axis=1, dtype=np.float64) / counts
                recomputed_scale = max(1e-8, float(np.sqrt(np.sum(weights * squares, dtype=np.float64) / 558)))
                self.require(recomputed_scale.hex() == float(self.reference_scale).hex(), 'historical R16 RMS exact reconstruction')
                self.near(self.reference_scale, max(1e-8, independent_rms), 'independent historical weighted RMS')
                for key, expected in self.reference.items():
                    value = self.data[f'r16_{key}']
                    self.require(value.dtype == expected.dtype and value.shape == expected.shape
                                 and value.tobytes() == expected.tobytes(), 'every original R16 target byte retained')
            else:
                self.near(self.data_info['r64_unfloored_rms_diagnostic_only'], independent_rms,
                          'R64 RMS is saved descriptively, not used to scale training')
            self.require(cast.dtype == np.float32 and cast.shape == (558, 4) and np.isfinite(cast).all()
                         and cast.tobytes() == (centered / self.reference_scale).astype(np.float32).tobytes(),
                         'float64 centering and fixed common scale precede training cast')
            for key, expected in (('weights_float64', weights), ('weights_float32', weights.astype(np.float32)),
                                  ('allowed', self.mask)):
                value = self.data[f'{kind}_{key}']
                self.require(value.dtype == expected.dtype and value.shape == expected.shape
                             and value.tobytes() == expected.tobytes(), 'exact common episode weights and masks')
            self.target_hashes[kind] = {key: hashlib.sha256(self.data[f'{kind}_{key}'].tobytes()).hexdigest()
                                       for key in ('centered_float64', 'scaled_float32', 'weights_float64', 'weights_float32', 'allowed')}
        self.counts.update(target_rows=558, target_episodes=144)

    def checkpoints(self):
        np = self.np
        self.checkpoint_desc = {}
        for seed in SEEDS:
            initials = []
            for kind in KINDS:
                for phase in ('initial', 'final'):
                    name = f'{kind}-{seed}-{phase}.npz'
                    with np.load(self.run / name, allow_pickle=False) as archive:
                        self.require(set(archive.files) == {'version', 'kind', 'input_dim', *SHAPES}, 'exact ordinary head schema')
                        self.require(archive['version'].item() == 'otto-symmetry-head-v1'
                                     and archive['kind'].item() == 'dense_augmented'
                                     and archive['input_dim'].item() == 2836, 'ordinary head metadata')
                        values = {key: archive[key] for key in SHAPES}
                    for key, value in values.items():
                        self.require(value.dtype == np.float32 and value.shape == SHAPES[key]
                                     and np.isfinite(value).all(), 'finite complete checkpoint arrays')
                    self.require(sum(value.size for value in values.values()) == 91380, 'head parameter count')
                    if phase == 'initial':
                        initials.append({k: v.tobytes() for k, v in values.items()})
                    self.checkpoint_desc[kind, seed, phase] = {'path': name, **self.worker['files'][name]}
            self.require(initials[0] == initials[1], 'byte-identical paired initialization')
        self.counts['checkpoints'] = 12

    def training(self):
        np = self.np
        self.checkpoints()
        events = iter(self.lines(self.run / 'training.jsonl', maximum=128 * 1024, count_limit=5895))
        counts, curves, parities = collections.Counter(), {}, {}
        event_count = 0

        def event(name, identity):
            nonlocal event_count
            value = next(events)
            event_count += 1
            self.require(value['event'] == name and all(value.get(k) == v for k, v in identity.items()),
                         'exact training event schedule')
            return value

        for seed in SEEDS:
            self.context = {'phase': 'training', 'seed': seed}
            operation_id = 0

            def operation(channel, identity, seed=seed):
                nonlocal operation_id
                context = {'seed': seed, 'operation_id': operation_id, 'channel': channel, **identity}
                self.equal(event('attempt', context), {'event': 'attempt', **context}, 'training attempt identity')
                returned = event('return', context)
                self.finite(returned['operation_seconds'], 'training operation seconds')
                if channel != 'optimizer_update':
                    self.require(returned['result'] is None, 'nonoptimizer return schema')
                counts[channel] += 1
                operation_id += 1
                return returned

            for kind in KINDS:
                declared = event('targets', {'seed': seed, 'arm': kind})
                self.equal(declared['arrays'], self.target_hashes[kind], 'training target array binding')
                self.equal(declared['scale'], self.data_info['scales'][kind], 'training target scale binding')
                operation('model_initialization', {'arm': kind})
                operation('checkpoint_export', {'arm': kind, 'phase': 'initial'})
            for kind in KINDS:
                operation('checkpoint_publication', {'arm': kind, 'phase': 'initial'})
            pair = event('initial_pair', {'seed': seed})
            self.require(pair['identical_weights'] is True, 'initial pair witness')
            self.equal(pair['initial_checkpoints'], {kind: self.checkpoint_desc[kind, seed, 'initial'] for kind in KINDS},
                       'paired initial descriptors')
            for kind in KINDS:
                self.equal(event('fit_start', {'seed': seed, 'arm': kind}),
                           {'event': 'fit_start', 'seed': seed, 'arm': kind, 'rows': 558}, 'fixed fit start')
                operation('optimizer_initialization', {'arm': kind})
                rng, epochs = np.random.default_rng(seed + 20000), []
                for epoch in range(1, 81):
                    declaration = event('order', {'seed': seed, 'arm': kind, 'epoch': epoch})
                    expected = rng.permutation(558).astype(np.int64)
                    self.equal(declaration['order'], expected.tolist(), 'paired seeded full-row order')
                    order_sha = hashlib.sha256(expected.tobytes()).hexdigest()
                    self.equal(declaration['sha256'], order_sha, 'epoch order byte hash')
                    losses, norms = [], []
                    for batch, offset in enumerate(range(0, 558, 128)):
                        rows = min(128, 558 - offset)
                        update = operation('optimizer_update', {'arm': kind, 'epoch': epoch,
                                           'batch': batch, 'offset': offset, 'rows': rows})['result']
                        self.require(update['rows'] == rows, 'actual partial-batch denominator')
                        self.finite(update['loss'], 'recorded weighted MSE')
                        self.finite(update['gradient_norm_before_clip'], 'recorded unclipped norm')
                        losses.append(update['loss'] * rows)
                        norms.append(update['gradient_norm_before_clip'])
                    curve = event('epoch', {'seed': seed, 'arm': kind, 'epoch': epoch})
                    self.require(curve['rows'] == 558 and curve['updates'] == 5 and curve['order_sha256'] == order_sha,
                                 'complete epoch coverage')
                    for key, expected_value in {'weighted_mse': math.fsum(losses) / 558,
                                                 'gradient_norm_mean': math.fsum(norms) / 5,
                                                 'gradient_norm_max': max(norms)}.items():
                        self.near(curve[key], expected_value, 'epoch reduction from recorded updates')
                    epochs.append({k: v for k, v in curve.items() if k != 'event'})
                for channel, extra in (('checkpoint_export', {'phase': 'final'}),
                                       ('checkpoint_publication', {'phase': 'final'}), ('inference_setup', {}),
                                       ('parity_numpy', {'rows': 16}), ('parity_torch', {'rows': 16})):
                    operation(channel, {'arm': kind, **extra})
                parity = event('parity', {'seed': seed, 'arm': kind})
                self.equal(parity['checkpoint'], self.checkpoint_desc[kind, seed, 'final'], 'parity final checkpoint')
                self.require(parity['rows'] == 16 and parity['passed'] is True
                             and parity['features_sha256'] == hashlib.sha256(self.data['features'][:16].tobytes()).hexdigest()
                             and parity['absolute_tolerance'] == parity['relative_tolerance'] == 2e-5, 'fixed TRAIN parity probe')
                left, right = np.asarray(parity['numpy'], dtype=np.float32), np.asarray(parity['torch'], dtype=np.float32)
                self.require(left.shape == right.shape == (16, 4) and np.isfinite(left).all() and np.isfinite(right).all()
                             and np.allclose(left, right, atol=2e-5, rtol=2e-5), 'recorded exported-weight parity')
                self.near(parity['maximum_difference'], float(np.max(np.abs(left.astype(np.float64) - right))), 'parity difference reduction')
                self.equal(event('fit_end', {'seed': seed, 'arm': kind}), {'event': 'fit_end', 'seed': seed, 'arm': kind,
                           'updates': 400, 'final_checkpoint': self.checkpoint_desc[kind, seed, 'final'], 'parity_passed': True},
                           'only fixed final fit admitted')
                curves[kind, seed], parities[kind, seed] = epochs, {k: v for k, v in parity.items() if k != 'event'}
            self.require(operation_id == 818, 'all pair operations')
        self.exhaust(events, 'no additional training work/events')
        self.require(event_count == 5895, 'exact training event count')
        expected_counts = {'model_initialization': 6, 'checkpoint_export': 12, 'checkpoint_publication': 12,
                           'optimizer_initialization': 6, 'optimizer_update': 2400, 'inference_setup': 6,
                           'parity_numpy': 6, 'parity_torch': 6}
        self.equal(dict(counts), expected_counts, 'complete training work')
        acknowledged = {key: {'attempted': value, 'returned': value} for key, value in counts.items()}
        self.equal(self.worker['training_calls'], acknowledged, 'worker training ledger')
        self.train_summary = self.read(self.run / 'training-summary.json')
        self.equal(self.train_summary['calls'], acknowledged, 'training summary ledger')
        self.require(self.train_summary['torch_threads'] == self.train_summary['torch_interop_threads'] == 1,
                     'original single-thread training witness')
        fits = iter(self.lines(self.run / 'fits.jsonl', maximum=256 * 1024, count_limit=6))
        parameter_names = {f'core.{layer}.{key}' for layer in (0, 2, 4) for key in ('weight', 'bias')}
        for seed in SEEDS:
            for kind in KINDS:
                fit = next(fits)
                self.require(fit['seed'] == seed and fit['arm'] == kind and fit['rows'] == 558, 'fixed fit identity')
                self.equal(fit['epochs'], curves[kind, seed], 'all 80 saved epoch curves')
                self.equal(fit['parity'], parities[kind, seed], 'fit parity record binding')
                self.equal(fit['optimizer_steps'], dict.fromkeys(parameter_names, 400), 'all Adam parameters reached step400')
                for phase in ('initial', 'final'):
                    self.equal(fit[f'{phase}_checkpoint'], self.checkpoint_desc[kind, seed, phase], 'fit checkpoint binding')
                self.equal(fit['costs'], self.train_summary['fits'][f'{kind}@{seed}'], 'fit timing witness')
                self.near(fit['costs']['label_allocation_seconds'], self.collection['seconds'] / 3 if kind == 'r64' else 0,
                          'incremental R48 label cost belongs to R64 family')
                self.near(fit['costs']['inherited_r16_label_allocation_seconds'], self.historical_collection['seconds'] / 3,
                          'both families disclose the same reused R16 label cost')
                self.near(fit['costs']['total_training_seconds'], fit['costs']['fit_seconds']
                          + fit['costs']['pair_setup_allocation_seconds'], 'complete allocated training time')
        self.exhaust(fits, 'exact six fits')
        self.counts.update(fits=6, epochs=480, optimizer_updates=2400, training_events=event_count)

    def evaluation(self):
        np = self.np
        self.deployment = self.read(self.run / 'deployment.json')
        self.require(set(self.deployment['heads']) == set(ARMS[:-1])
                     and self.deployment['head_allocation_episodes'] == 72
                     and self.deployment['module_allocation_episodes'] == 432, 'deployment coverage/allocation')
        for arm in ARMS[:-1]:
            kind, seed = arm.split('@')
            self.equal(self.deployment['heads'][arm]['checkpoint'], self.checkpoint_desc[kind, int(seed), 'final'],
                       'deployed fixed final checkpoint')
            self.finite(self.deployment['heads'][arm]['seconds'], 'checkpoint load interval')
        self.finite(self.deployment['model_module_seconds'], 'deployment module interval')
        rows = iter(self.lines(self.run / 'evaluation.jsonl', maximum=1024 * 1024, count_limit=504))
        transitions = iter(self.lines(self.run / 'eval-transitions.jsonl.gz', maximum=8192, count_limit=504 * (H + 1)))
        work = iter(self.lines(self.run / 'work.jsonl', maximum=2048, count_limit=2 * (1008 + 3 * 504 * H)))
        counts, times = collections.Counter(), collections.defaultdict(list)
        self.eval_rows = []
        current_case, paired_source, paired_draws = None, None, {}

        def operation(name, episode, step):
            identity = {'operation': name, 'index': counts[name], 'phase': 'evaluation', 'episode_id': episode, 'step': step}
            self.equal(next(work), {'event': 'attempt', **identity}, 'evaluation operation attempt')
            returned = next(work)
            self.require(returned['event'] == 'return' and all(returned[k] == v for k, v in identity.items()), 'joined evaluation return')
            self.finite(returned['seconds'], 'operation timing witness')
            counts[name] += 1
            times[name].append(returned['seconds'])
            return returned['seconds']

        for regime, seed, case, hit, block, arm in schedule():
            self.context = {'phase': 'evaluation', 'regime': regime, 'seed': seed, 'arm': arm}
            row = next(rows)
            episode = f'eval:{regime}:{seed}:{arm}'
            identity = {'episode_id': episode, 'regime': regime, 'seed': seed, 'case': case,
                        'initial_hit': hit, 'block': block, 'arm': arm}
            self.require(all(row[k] == v for k, v in identity.items()), 'exact rotated evaluation schedule')
            self.require(type(row['steps']) is int and 1 <= row['steps'] <= H and type(row['found']) is bool
                         and (row['found'] or row['steps'] == H) and row['updates'] == row['steps']
                         and row['blocked_steps'] == 0 and row['final_update_assimilated'] is True, 'honest completed evaluation')
            reset = next(transitions)
            self.require(reset['kind'] == 'reset' and all(reset[k] == v for k, v in identity.items()), 'reset identity')
            public = self.packet(reset['public'], 0)
            self.require(public['position'] == [26, 26] and public['hit'] == hit and not public['done'], 'positive-hit center reset')
            source = reset['source_evaluation_only']
            self.equal(row['source_evaluation_only'], source, 'episode source witness')
            self.require(len(source) == 2 and all(type(v) is int and 0 <= v < 53 for v in source), 'source coordinate witness')
            state = reset['posterior_after']
            self.near(operation('native_reset', episode, 0), row['environment_reset_seconds'], 'native reset cost')
            self.near(operation('actor_initialization', episode, 0), row['init_seconds'], 'all actor initialization charged')
            sums = collections.defaultdict(list)
            for step in range(1, row['steps'] + 1):
                record = next(transitions)
                self.require(record['kind'] == 'step' and record['episode_id'] == episode and record['step'] == step,
                             'complete sequential evaluation transition')
                self.equal(record['posterior_before'], state, 'linked posterior witnesses')
                self.equal(record['allowed_actions'], public['valid_actions'], 'decision eligibility')
                action = self.saved_choice(record['scores'], public['valid_actions'])
                self.require(record['action'] == action, 'deployed action matches recorded near-tie rule')
                operation('analytic_choose' if arm == 'analytic_inbounds' else 'head_predict', episode, step)
                self.near(operation('native_step', episode, step), record['environment_seconds'], 'native move cost')
                self.near(operation('public_update', episode, step), record['update_seconds'], 'final public update cost')
                position = list(public['position'])
                position[action // 2] += 2 * (action % 2) - 1
                after = self.packet(record['public'], step)
                self.equal(after['position'], position, 'actual inbounds movement')
                self.require(after['done'] == (position == source) and (not after['done'] or step == row['steps']), 'first source discovery')
                self.near(record['choose_seconds'], record['choose_instrumented_seconds'] - record['excluded_io_seconds'],
                          'choice removes only recorded journal time')
                for key in ('choose_seconds', 'choose_instrumented_seconds', 'excluded_io_seconds', 'update_seconds', 'environment_seconds'):
                    self.finite(record[key], 'nonnegative complete transition interval')
                    sums[key].append(record[key])
                self.finite(record['native_p_end'], 'recorded native p_end')
                self.require(set(record['posterior_after']) == {'sha256', 'mass'}
                             and len(record['posterior_after']['sha256']) == 64, 'posterior witness schema')
                self.finite(record['posterior_after']['mass'], 'posterior mass witness')
                public, state = after, record['posterior_after']
            self.equal(row['final_public'], public, 'final packet retained')
            self.require(public['done'] == row['found'], 'final outcome identity')
            if row['found']:
                point = np.zeros((53, 53), dtype=np.float64)
                point[tuple(public['position'])] = 1
                self.equal(state, {'mass': 1.0, 'sha256': hashlib.sha256(point.tobytes()).hexdigest()}, 'terminal point mass witness')
            for key, values in sums.items():
                self.near(row[key], math.fsum(values), 'complete episode timing sum', rtol=1e-9, atol=1e-9)
            allocation = (self.deployment['heads'][arm]['seconds'] / 72 + self.deployment['model_module_seconds'] / 432
                          if arm != 'analytic_inbounds' else 0)
            self.near(row['setup_allocation_seconds'], allocation, 'complete deployment allocation')
            self.near(row['controller_seconds'], math.fsum(row[k] for k in
                      ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')), 'controller cost accounting')
            if current_case != (regime, case):
                current_case, paired_source, paired_draws = (regime, case), source, {}
            self.equal(source, paired_source, 'paired case source')
            draws = row['draws_evaluation_only']
            expected_hits = row['steps'] - int(row['found'])
            self.equal([(d['channel'], d['index']) for d in draws], [('source', 0)] + [('hit', i) for i in range(expected_hits)],
                       'no odor draw upon finding')
            generators = {channel: np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, number])))
                          for channel, number in (('source', 1), ('hit', 2))}
            for draw in draws:
                key = draw['channel'], draw['index']
                self.equal(draw['uniform'], float(generators[key[0]].random()), 'native seeded uniform witness')
                if key in paired_draws:
                    self.equal(draw['uniform'], paired_draws[key], 'paired uniform across all overlapping arm paths')
                paired_draws[key] = draw['uniform']
                self.near(draw['cdf_mass'], 1, 'native CDF normalization', atol=1e-10)
                self.require(type(draw['selected_index']) is int and 0 <= draw['selected_index'] < (2809 if key[0] == 'source' else 4),
                             'native categorical index range')
                if key[0] == 'source':
                    self.equal(list(divmod(draw['selected_index'], 53)), source, 'native source index')
            self.eval_rows.append({k: v for k, v in row.items() if k != 'draws_evaluation_only'})
        self.exhaust(rows, 'exact 504 evaluation episodes')
        self.exhaust(transitions, 'no surplus evaluation transitions')
        self.exhaust(work, 'no surplus native/controller work')
        self.require(counts['native_reset'] == counts['actor_initialization'] == 504
                     and counts['native_step'] == counts['public_update'] == counts['head_predict'] + counts['analytic_choose'],
                     'all evaluation operations accounted')
        self.require(set(counts) == set(self.worker['calls']), 'exact work channels')
        for name, count in counts.items():
            record = self.worker['calls'][name]
            self.require(record['attempted'] == record['returned'] == count, 'worker work counts')
            self.near(record['seconds'], math.fsum(times[name]), 'worker work timing sum', rtol=1e-9, atol=1e-8)
        self.counts.update(evaluation_episodes=504, evaluation_cases=72, evaluation_moves=counts['native_step'])

    def tree(self, actual, expected, label):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), label)
            for key, value in expected.items():
                self.tree(actual[key], value, f'{label}.{key}')
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label)
            for left, right in zip(actual, expected, strict=True):
                self.tree(left, right, label)
        elif type(expected) is float:
            self.near(actual, expected, label, rtol=1e-10, atol=1e-10)
        else:
            self.equal(actual, expected, label)

    def conclusions(self):
        self.context = {'phase': 'independent criteria'}
        summary = self.read(self.run / 'summary.json')
        metrics = ('found', 'steps', 'init_seconds', 'choose_seconds', 'update_seconds',
                   'setup_allocation_seconds', 'controller_seconds', 'environment_seconds')
        positive, competence, relative_checks, regimes = [], [], [], {}

        def criterion(destination, name, actual, bound, passes):
            destination.append({'name': name, 'value': actual, 'threshold': bound, 'passes': bool(passes)})

        for regime in FIRST:
            local = [r for r in self.eval_rows if r['regime'] == regime]
            weights = self.mixtures[regime]

            def weighted(arm, metric, block=None, local=local, weights=weights):
                subset = [r for r in local if r['arm'] == arm and (block is None or r['block'] == block)]
                expected_size = 8 if block is None else 1
                terms = []
                for hit in (1, 2, 3):
                    values = [float(row[metric]) for row in subset if row['initial_hit'] == hit]
                    self.require(len(values) == expected_size, 'complete mixture stratum denominator')
                    terms.append(weights[hit] * math.fsum(values) / expected_size)
                return math.fsum(terms)

            means = {arm: {key: weighted(arm, key) for key in metrics} for arm in ARMS}
            families = {kind: {key: math.fsum(means[f'{kind}@{seed}'][key] for seed in SEEDS) / 3
                               for key in metrics} for kind in KINDS}
            blocks = [{arm: weighted(arm, 'steps', block) for arm in ARMS} for block in range(8)]
            control = means['analytic_inbounds']
            criterion(positive, f'{regime}.analytic_control.success', control['found'], .95, control['found'] >= .95)
            for seed in SEEDS:
                value = means[f'r64@{seed}']
                criterion(competence, f'{regime}.{seed}.success', value['found'], .95, value['found'] >= .95)
                criterion(competence, f'{regime}.{seed}.moves', value['steps'], 1.05 * control['steps'],
                          value['steps'] <= 1.05 * control['steps'])
            candidate, baseline = families['r64'], families['r16']
            gains = [math.fsum(block[f'r16@{seed}'] - block[f'r64@{seed}'] for seed in SEEDS) / 3
                     for block in blocks]
            wins = sum(value > 0 for value in gains)
            criterion(relative_checks, f'{regime}.success', candidate['found'], baseline['found'], candidate['found'] >= baseline['found'])
            criterion(relative_checks, f'{regime}.moves', candidate['steps'], .95 * baseline['steps'], candidate['steps'] <= .95 * baseline['steps'])
            criterion(relative_checks, f'{regime}.positive_blocks', wins, 6, wins >= 6)
            criterion(relative_checks, f'{regime}.controller_cost', candidate['controller_seconds'], 1.05 * baseline['controller_seconds'],
                      candidate['controller_seconds'] <= 1.05 * baseline['controller_seconds'])
            regimes[regime] = {'weights': {str(h): weights[h] for h in (1, 2, 3)},
                'means': means, 'family_means': families, 'blocks': blocks, 'paired_family_block_gains': gains,
                'strata': {str(h): {arm: {key: math.fsum(float(r[key]) for r in local if r['arm'] == arm
                           and r['initial_hit'] == h) / 8 for key in metrics} for arm in ARMS} for h in (1, 2, 3)},
                'raw_counts': {arm: {'found': sum(r['found'] for r in local if r['arm'] == arm), 'episodes': 24} for arm in ARMS}}
        self.require((len(positive), len(competence), len(relative_checks)) == (3, 18, 12), 'all 33 independent criteria')
        passed = all(row['passes'] for row in positive + competence + relative_checks)
        expected = {'regimes': regimes, 'positive_control_checks': positive, 'competence_checks': competence,
                    'relative_checks': relative_checks, 'pilot_continuation': passed,
                    'episodes': 504, 'paired_cases': 72, 'architecture_advantage_established': False,
                    'inherited_results_revised': False, 'version': 'otto-target-precision-v1'}
        self.tree({key: summary[key] for key in expected}, expected, 'independent study conclusion')
        self.equal(self.worker['pilot_continuation'], passed, 'worker criterion result')
        for value in summary['costs'].values():
            self.finite(value, 'physical stage interval')
        self.near(summary['physical_stage_seconds'], math.fsum(summary['costs'].values()), 'disjoint measured stage subtotal')
        self.require(summary['physical_stage_seconds'] <= self.worker['wall_seconds'] + 1e-6,
                     'stage subtotal fits complete worker time')
        self.near(summary['costs']['additional_r48_collection_seconds'], self.collection['seconds'], 'complete collection interval')
        self.near(summary['costs']['training_data_seconds'], self.data_info['seconds'], 'target preparation interval')
        self.near(summary['costs']['model_module_seconds'], self.deployment['model_module_seconds'], 'deployment module interval')
        self.near(summary['costs']['head_restore_seconds'], math.fsum(v['seconds'] for v in self.deployment['heads'].values()),
                  'six actual checkpoint restores')
        self.near(summary['costs']['training_pairs_seconds'], math.fsum(p['seconds'] for p in self.train_summary['pairs']),
                  'three complete training pairs')
        self.equal(summary['fit_costs'], self.train_summary['fits'], 'summary fit costs')
        self.equal([p['seed'] for p in self.train_summary['pairs']], list(SEEDS), 'all paired fitting intervals')
        for pair in self.train_summary['pairs']:
            seed = pair['seed']
            fit_sum = math.fsum(self.train_summary['fits'][f'{kind}@{seed}']['fit_seconds'] for kind in KINDS)
            self.near(pair['nonfit_seconds'], pair['seconds'] - fit_sum, 'pair setup cost separated')
            for kind in KINDS:
                self.near(self.train_summary['fits'][f'{kind}@{seed}']['pair_setup_allocation_seconds'], pair['nonfit_seconds'] / 2,
                          'common pairing overhead allocation')
        self.finite(summary['measured_journal_io_seconds'], 'partial overlapping journal timer')
        self.result = {**expected, 'version': VERSION, 'producer_version': 'otto-target-precision-v1',
                       'criteria_passed': sum(r['passes'] for r in positive + competence + relative_checks),
                       'criteria_total': 33, 'collection': {k: self.collection[k] for k in
                           ('panels', 'records', 'steps', 'found', 'censored', 'sampler_events', 'seconds')},
                       'costs': summary['costs'], 'limitations': LIMITATIONS,
                       'precision_labels': {'reused_records': 32304, 'new_records': 96912, 'combined_records': 129216,
                           'replicate_ids_new': list(range(16, 64)), 'shared_original_r16_scale': self.reference_scale,
                           'r64_unfloored_rms_diagnostic_only': self.data_info['r64_unfloored_rms_diagnostic_only'],
                           'historical_r16_collection_seconds': self.historical_collection['seconds']},
                       'timing_scope': 'Stage intervals and journal costs are saved witnesses; process wall includes final closure.',
                       'scientific_gate_is_not_technical_completion': True}

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupted(_signal, _frame):
            raise InterruptedError('saved-audit supervision interrupted execution')

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admission()
            self.authenticate()
            self.arrays()
            self.panels()
            self.targets()
            self.training()
            self.evaluation()
            self.conclusions()
            self.context = {'phase': 'closing immutable evidence'}
            for name, desc in self.bound.items():
                self.equal(digest(Path(name), self.check), desc, 'unchanged evidence at audit close')
            self.require(digest(relative(SELF), self.check) == self.self_source, 'unchanged auditor source')
            write(self.out / 'audit.json', self.result)
            self.check()
            files = {p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()}
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, counts=dict(self.counts),
                audit_plan_sha256=self.args.audit_plan_sha256, source=self.self_source,
                clock_sha256=CLOCK_PIN, clock_backend=self.clock.backend,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                files=files, criteria_passed=self.result['criteria_passed'], criteria_total=33,
                pilot_continuation=self.result['pilot_continuation'],
                timing_scope='Through audit payload hashing; original parent covers receipt publication and exit.',
                count_scope='Through payload hashing; post-publication resource checks are excluded.')
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'agreement': True,
                              'receipt_sha256': digest(self.out / 'receipt.json')['sha256']}), flush=True)
        except BaseException as error:
            self.receipt.update(status='failed', agreement=False, counts=dict(self.counts),
                failures=[{'error': repr(error), 'context': self.context, 'traceback': traceback.format_exc()}])
            try:
                self.receipt['wall_seconds'] = ((self.clock.now_ns() - self.start) / 1e9
                                               if self.clock and self.start is not None else None)
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure
                self.receipt['clock_error'] = repr(secondary)
            try:
                target = self.out / 'receipt.json'
                if target.exists():
                    target.rename(self.out / 'receipt.invalid.json')
                self.receipt['files'] = {p.name: digest(p) for p in self.out.iterdir() if p.is_file()}
                write(target, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original publication failure
                error.add_note(f'Failure receipt publication error: {secondary!r}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-plan', type=Path, required=True)
    parser.add_argument('--audit-plan-sha256', required=True)
    parser.add_argument('--supervision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    regular(args.audit_plan)
    regular(args.supervision)
    regular(args.output)
    Audit(args).execute()


if __name__ == '__main__':
    main()
