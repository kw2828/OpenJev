"""Independent saved-record audit of fixed 80-versus-320-epoch training.

Requires a separately frozen external audit plan and original supervisor.
Uses saved-array arithmetic and deterministic RNG reconstruction only. It never
imports or calls a scientific producer, model, optimizer, sampler or simulator.
The prior independently audited R64 cache is inherited without label replay.
New work checks the exact matched prefix, all fixed-final TRAIN score reductions,
the complete fresh evaluation ledger and all 33 unchanged continuation rules.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-training-budget-saved-audit-v1'
SELF = 'scripts/audit_otto_training_budget.py'
BASE_AUDITOR = 'scripts/audit_otto_target_precision.py'
BASE_PIN = '9f8dbc193d72a7791aa6e87dfcc817bf6351edcd638dd70cf911935061792071'
PRESERVED = {
    'output/otto-target-precision-v1/plan-01.json': '921318b9c978b208989ad66210a06adad8f104c827351eda1850f080d9f8a46a',
    'output/otto-target-precision-v1/run-01/receipt.json': '34b331507fe8f53058baecdbeed0e3448b91202a6971762949d1ae7017f4bb2a',
    'output/otto-target-precision-v1/supervision-01.terminal.json': 'fc3312338d04b5a0955727651e83a62f5907cedff7be007e165d4f4637a71e37',
    'output/otto-target-precision-v1/audit-01/receipt.json': '549eb00f8a04904f22bb9ff351c35c4e5f965b88a6f1944d0186886ac693baa7',
    'output/otto-target-precision-v1/audit-supervision-01.terminal.json': 'bf756bbb82639b78de692bf7bc0cba818306ce82d7ca599c99d68ff45b272a34',
}
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
SUPERVISOR_PIN = '610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144'
PRODUCER = 'scripts/study_otto_training_budget.py'
OLD = 'output/otto-target-precision-v1/run-01'
PLAN_PIN = 'a204495fb0fd66ea3e7ea7c5052315f85bf3fa75438c243cf6a6f17e92e4099a'
SYMM = 'output/otto-symmetry-head-v1/run-01'
LIMITS = {'seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 64 * 1024**2}
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
H, SEEDS = 2188, (30101, 30102, 30103)
KINDS = ('short', 'long')
EPOCHS = {'short': 80, 'long': 320}
ARMS = tuple(f'{kind}@{seed}' for seed in SEEDS for kind in KINDS) + ('analytic_inbounds',)
FIRST = {'lambda3': 1080001, 'lambda4': 1090001, 'lambda5': 1100001}
ROOT_FILES = {'started.json', 'training-data.npz', 'training-data.json', 'training.jsonl',
              'fits.jsonl', 'training-summary.json', 'diagnostics.jsonl', 'diagnostic-summary.json',
              'deployment.json', 'evaluation.jsonl', 'eval-transitions.jsonl.gz', 'work.jsonl', 'summary.json'}
PAYLOADS = ROOT_FILES | {
    f'{kind}-{seed}-{phase}.npz' for kind in KINDS for seed in SEEDS for phase in ('initial', 'final', 'diagnostics')
} | {f'long-{seed}-prefix.npz' for seed in SEEDS}
SHAPES = {'weight0': (32, 2836), 'bias0': (32,), 'weight1': (16, 32), 'bias1': (16,),
          'weight2': (4, 16), 'bias2': (4,)}
LIMITATIONS = [
    'Saved-record audit only: zero fresh teacher, sampler, model, optimizer or simulator calls.',
    'Original R64 labels, feature construction and filtering inherit the authenticated prior completed audit.',
    'Optimizer execution, gradients, neural scores and feature transformations are not regenerated.',
    'All new checkpoint bytes, recorded first-16 TRAIN parity and exact shared training prefix are checked.',
    'Fixed-final TRAIN diagnostics are independently reduced from saved float32 scores; labels are sampled costs, not true regret.',
    'Evaluation choices use saved scores and seeded draw witnesses; probability vectors and intermediate beliefs are not regenerated.',
    'Evaluation posterior witnesses are linked and terminal point masses checked, without fresh filtering.',
    'Timing, actual native execution and durable publication remain authenticated original-process evidence.',
    'Historical failures and successful audits are retained and authenticated, not rerun or revised.',
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
        self.equal(plan['preserved_inputs'], PRESERVED, 'fixed completed precision study and audit lineage')
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
            self.require(self.bind(relative(name))['sha256'] == pin, 'unchanged inherited evidence')
        self.receipt.update(base_auditor_sha256=BASE_PIN, preserved_inputs=PRESERVED,
                            scope='Unchanged audited R64 cache, exact training prefix, saved final TRAIN diagnostics and full fresh evaluation.')
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
        self.require(plan['status'] == 'frozen_before_training' and worker['status'] == 'completed',
                     'completed producer required; failed-study review needs separate admission')
        self.require(plan['version'] == worker['version'] == 'otto-training-budget-v1'
                     and worker['plan_sha256'] == PLAN_PIN and worker['requires_successful_original_supervisor'] is True,
                     'producer identity')
        self.require(worker['pending'] == [] and worker['pending_episode'] is None,
                     'producer work closure')
        self.equal([worker[k] for k in ('completed_diagnostics', 'completed_fits', 'completed_episodes')],
                   [6, 6, 504], 'complete scientific stages')
        self.equal(plan['limits'], {'native_seconds': 7200, 'rss_bytes': 4 * 1024**3,
                                   'output_bytes': 16 * 1024**3}, 'producer budgets')
        self.require(set(worker['files']) == PAYLOADS
                     and {p.name for p in self.run.iterdir()} == PAYLOADS | {'receipt.json'}, 'exact 34 payload closure')
        for name, desc in worker['files'].items():
            self.bind(self.run / name, desc)
        for name, pin in plan['sources'].items():
            self.require(self.bind(relative(name))['sha256'] == pin, 'frozen producer source')
        for name, desc in plan['inputs'].items():
            self.bind(relative(name), desc)
        self.require(len(plan['sources']) == 164 and len(plan['inputs']) == 1748, 'complete training-budget manifest')
        self.counts.update(payloads=34, sources=len(plan['sources']), inputs=len(plan['inputs']))
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
                     and launch['cap_seconds'] == 7200 and launch['clock_source_sha256'] == CLOCK_PIN
                     and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
                     and launch['clock_backend'] == worker['clock_backend']
                     and launch['deadline_ns'] == launch['started_ns'] + 7200 * 10**9,
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
        historical_audit = self.read(relative('output/otto-target-precision-v1/audit-01/receipt.json'))
        self.require(historical_worker['status'] == historical_audit['status'] == 'completed'
                     and historical_audit['agreement'] is True
                     and historical_audit['version'] == 'otto-target-precision-saved-audit-v1',
                     'original R64 provenance inherits its completed independent saved audit')
        self.equal(historical_audit['source'], self.bound[str(relative(BASE_AUDITOR))], 'original completed auditor source')
        self.require(historical_worker['completed_panels'] == 558 and historical_worker['completed_fits'] == 6
                     and historical_worker['completed_episodes'] == 504, 'complete original study is preserved')
        self.receipt.update(producer_inputs=entries, producer_parent_seconds=terminal['wall_seconds'])

    def arrays(self):
        import numpy as np
        self.np = np
        self.require(np.__version__ == self.plan['all_distributions']['numpy'], 'qualified saved-array version')
        self.mixtures = {}
        for regime in FIRST:
            with np.load(relative(f'{SYMM}/kernel-{regime}.npz'), allow_pickle=False) as saved:
                self.require(set(saved.files) == {'likelihood', 'initial_hit_weights'}, 'public kernel schema')
                mixture = saved['initial_hit_weights']
            self.require(mixture.dtype == np.float64 and mixture.shape == (4,) and mixture[0] == 0
                         and np.isfinite(mixture).all() and (mixture[1:] > 0).all(), 'positive hit mixture')
            self.near(float(mixture.sum()), 1, 'hit mixture normalization')
            self.mixtures[regime] = mixture.tolist()
        for row in self.rows:
            self.packet(row['public'], row['prefix_index'])
        with np.load(self.run / 'training-data.npz', allow_pickle=False) as saved:
            self.data = {key: saved[key] for key in saved.files}
        fields = ('centered_float64', 'scaled_float32', 'weights_float64', 'weights_float32', 'allowed')
        expected = {'features', 'r64_costs', 'allowed', 'anchor_ids'} | {
            f'{kind}_{key}' for kind in KINDS for key in fields}
        self.require(set(self.data) == expected, 'exact 14 TRAIN arrays')
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
                   'original audited R64 cache descriptor')
        with np.load(relative(cache_name), allow_pickle=False) as saved:
            inherited = {'features', 'r16_costs', 'r64_costs', 'allowed', 'anchor_ids'} | {
                f'{kind}_{key}' for kind in ('r16', 'r64') for key in fields}
            self.require(set(saved.files) == inherited, 'inherited precision cache schema')
            self.reference = {key: saved[f'r64_{key}'] for key in fields}
            for key in ('features', 'r64_costs', 'allowed', 'anchor_ids'):
                self.require(self.data[key].dtype == saved[key].dtype and self.data[key].shape == saved[key].shape
                             and self.data[key].tobytes() == saved[key].tobytes(), 'unchanged original features/costs/mask/order')
        original_info = self.read(relative(f'{OLD}/training-data.json'))
        self.equal(original_info['rows'], self.rows, 'unchanged original row selection')
        self.reference_scale = original_info['scales']['r64']
        self.finite(self.reference_scale, 'inherited R64 scale')
        self.require(self.reference_scale >= 1e-8, 'original scale floor')
        historical = self.read(relative(f'{OLD}/collection.json'))
        self.inherited_label_seconds = historical['seconds'] + historical['historical_r16_collection_seconds']

    def targets(self):
        np = self.np
        costs = self.data['r64_costs']
        self.require(costs.dtype == np.float64 and costs.shape == (558, 4)
                     and np.isfinite(costs[self.mask]).all() and np.isposinf(costs[~self.mask]).all(),
                     'finite eligible and explicitly blocked original costs')
        episodes = collections.Counter(row['episode_id'] for row in self.rows)
        weights = np.asarray([558 / (144 * episodes[row['episode_id']]) for row in self.rows], dtype=np.float64)
        schemas = {'centered_float64': (np.float64, (558, 4)), 'scaled_float32': (np.float32, (558, 4)),
                   'weights_float64': (np.float64, (558,)), 'weights_float32': (np.float32, (558,)),
                   'allowed': (np.bool_, (558, 4))}
        hashes = {}
        for key, (dtype, shape) in schemas.items():
            reference = self.reference[key]
            self.require(reference.dtype == dtype and reference.shape == shape
                         and np.isfinite(reference).all(), 'finite typed inherited target')
            hashes[key] = hashlib.sha256(reference.tobytes()).hexdigest()
        self.equal(self.data_info['reference'], {'scale': self.reference_scale, 'arrays': hashes}, 'exact reference binding')
        self.require(np.array_equal(self.reference['allowed'], self.mask)
                     and self.reference['weights_float64'].tobytes() == weights.tobytes()
                     and self.reference['weights_float32'].tobytes() == weights.astype(np.float32).tobytes(),
                     'unchanged equal-episode weights and public mask')
        self.require((self.reference['centered_float64'][~self.mask] == 0).all()
                     and (self.reference['scaled_float32'][~self.mask] == 0).all(), 'blocked target zero')
        self.target_hashes = {}
        for kind in KINDS:
            self.require(type(self.data_info['scales'][kind]) is float
                         and self.data_info['scales'][kind].hex() == self.reference_scale.hex(), 'unchanged scale bits')
            for key in schemas:
                actual, expected = self.data[f'{kind}_{key}'], self.reference[key]
                self.require(actual.dtype == expected.dtype and actual.shape == expected.shape
                             and actual.tobytes() == expected.tobytes(), 'both arms preserve all original R64 target bytes')
            self.target_hashes[kind] = dict(hashes)
        self.counts.update(target_rows=558, target_episodes=144, new_teacher_labels=0)

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

    def saved_choice(self, scores, allowed):
        self.require(len(scores) == 4 and allowed, 'four-action decision')
        for action in allowed:
            self.finite(scores[action], 'eligible saved score', nonnegative=False)
        minimum = min(scores[a] for a in allowed)
        return next(a for a in allowed if abs(float(scores[a]) - minimum) < 1e-10)

    def checkpoints(self):
        np = self.np
        self.checkpoint_desc, fingerprints = {}, {}
        for seed in SEEDS:
            initials = []
            for kind in KINDS:
                for phase in (('initial', 'final', 'prefix') if kind == 'long' else ('initial', 'final')):
                    name = f'{kind}-{seed}-{phase}.npz'
                    with np.load(self.run / name, allow_pickle=False) as archive:
                        self.require(set(archive.files) == {'version', 'kind', 'input_dim', *SHAPES}, 'exact ordinary head schema')
                        self.require(archive['version'].item() == 'otto-symmetry-head-v1'
                                     and archive['kind'].item() == 'dense_augmented'
                                     and archive['input_dim'].item() == 2836, 'ordinary head metadata')
                        values = {key: archive[key] for key in SHAPES}
                        fingerprints[kind, seed, phase] = {key: (archive[key].dtype.str, archive[key].shape, archive[key].tobytes())
                                                          for key in archive.files}
                    for key, value in values.items():
                        self.require(value.dtype == np.float32 and value.shape == SHAPES[key]
                                     and np.isfinite(value).all(), 'finite complete checkpoint arrays')
                    self.require(sum(value.size for value in values.values()) == 91380, 'head parameter count')
                    if phase == 'initial':
                        initials.append({k: v.tobytes() for k, v in values.items()})
                    self.checkpoint_desc[kind, seed, phase] = {'path': name, **self.worker['files'][name]}
            self.require(initials[0] == initials[1], 'byte-identical paired initialization')
            self.require(fingerprints['short', seed, 'final'] == fingerprints['long', seed, 'prefix'],
                         'all long-prefix checkpoint bytes exactly equal paired short final')
        self.counts['checkpoints'] = 15

    def training(self):
        np = self.np
        self.checkpoints()
        events = iter(self.lines(self.run / 'training.jsonl', maximum=128 * 1024, count_limit=14550))
        counts, curves, parities = collections.Counter(), {}, {}
        self.prefix_witnesses = []
        parameter_names = {f'core.{layer}.{key}' for layer in (0, 2, 4) for key in ('weight', 'bias')}
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
            short_orders, short_batches, short_epochs = [], [], []

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
                orders, batches = [], []
                for epoch in range(1, EPOCHS[kind] + 1):
                    declaration = event('order', {'seed': seed, 'arm': kind, 'epoch': epoch})
                    expected = rng.permutation(558).astype(np.int64)
                    self.equal(declaration['order'], expected.tolist(), 'paired seeded full-row order')
                    order_sha = hashlib.sha256(expected.tobytes()).hexdigest()
                    self.equal(declaration['sha256'], order_sha, 'epoch order byte hash')
                    orders.append(declaration['order'])
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
                        batches.append({'epoch': epoch, 'batch': batch, 'offset': offset, **update})
                    curve = event('epoch', {'seed': seed, 'arm': kind, 'epoch': epoch})
                    self.require(curve['rows'] == 558 and curve['updates'] == 5 and curve['order_sha256'] == order_sha,
                                 'complete epoch coverage')
                    for key, expected_value in {'weighted_mse': math.fsum(losses) / 558,
                                                 'gradient_norm_mean': math.fsum(norms) / 5,
                                                 'gradient_norm_max': max(norms)}.items():
                        self.near(curve[key], expected_value, 'epoch reduction from recorded updates')
                    epochs.append({k: v for k, v in curve.items() if k != 'event'})
                    if kind == 'long' and epoch == 80:
                        self.equal(orders, short_orders, 'exact first 80 paired epoch orders')
                        self.equal(batches, short_batches, 'exact first 400 paired losses and gradient norms')
                        self.equal([{k: v for k, v in row.items() if k != 'arm'} for row in epochs],
                                   short_epochs, 'exact first 80 epoch reductions')
                        operation('checkpoint_export', {'arm': kind, 'phase': 'prefix'})
                        operation('checkpoint_publication', {'arm': kind, 'phase': 'prefix'})
                        prefix = event('matched_prefix', {'seed': seed, 'arm': kind, 'epoch': 80})
                        expected_prefix = {'event': 'matched_prefix', 'seed': seed, 'arm': kind, 'epoch': 80,
                            'updates': 400, 'checkpoint': self.checkpoint_desc[kind, seed, 'prefix'],
                            'short_final_checkpoint': self.checkpoint_desc['short', seed, 'final'],
                            'identical_weights': True, 'identical_order_records': True,
                            'identical_update_records': True, 'identical_epoch_records': True,
                            'compared_epochs': 80, 'compared_updates': 400,
                            'optimizer_steps': dict.fromkeys(parameter_names, 400)}
                        self.equal(prefix, expected_prefix, 'durably published matching prefix before more updates')
                        self.prefix_witnesses.append({k: v for k, v in prefix.items() if k != 'event'})
                if kind == 'short':
                    short_orders, short_batches = orders, batches
                    short_epochs = [{k: v for k, v in row.items() if k != 'arm'} for row in epochs]
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
                           'updates': 5 * EPOCHS[kind], 'final_checkpoint': self.checkpoint_desc[kind, seed, 'final'], 'parity_passed': True},
                           'only fixed final fit admitted')
                curves[kind, seed], parities[kind, seed] = epochs, {k: v for k, v in parity.items() if k != 'event'}
            self.require(operation_id == 2020, 'all pair operations')
        self.exhaust(events, 'no additional training work/events')
        self.require(event_count == 14550, 'exact training event count')
        expected_counts = {'model_initialization': 6, 'checkpoint_export': 15, 'checkpoint_publication': 15,
                           'optimizer_initialization': 6, 'optimizer_update': 6000, 'inference_setup': 6,
                           'parity_numpy': 6, 'parity_torch': 6}
        self.equal(dict(counts), expected_counts, 'complete training work')
        acknowledged = {key: {'attempted': value, 'returned': value} for key, value in counts.items()}
        self.equal(self.worker['training_calls'], acknowledged, 'worker training ledger')
        self.train_summary = self.read(self.run / 'training-summary.json')
        self.equal(self.train_summary['calls'], acknowledged, 'training summary ledger')
        self.require(self.train_summary['torch_threads'] == self.train_summary['torch_interop_threads'] == 1,
                     'original single-thread training witness')
        fits = iter(self.lines(self.run / 'fits.jsonl', maximum=256 * 1024, count_limit=6))
        for seed in SEEDS:
            for kind in KINDS:
                fit = next(fits)
                self.require(fit['seed'] == seed and fit['arm'] == kind and fit['rows'] == 558, 'fixed fit identity')
                self.equal(fit['epochs'], curves[kind, seed], 'all fixed-budget saved epoch curves')
                self.equal(fit['parity'], parities[kind, seed], 'fit parity record binding')
                self.equal(fit['optimizer_steps'], dict.fromkeys(parameter_names, 5 * EPOCHS[kind]), 'all Adam parameters reached declared final step')
                for phase in ('initial', 'final'):
                    self.equal(fit[f'{phase}_checkpoint'], self.checkpoint_desc[kind, seed, phase], 'fit checkpoint binding')
                self.equal(fit['costs'], self.train_summary['fits'][f'{kind}@{seed}'], 'fit timing witness')
                self.equal(fit['costs']['label_allocation_seconds'], 0., 'zero new label generation')
                self.near(fit['costs']['inherited_r64_label_allocation_seconds'], self.inherited_label_seconds / 3,
                          'both families disclose the same inherited full R64 label cost')
                self.near(fit['costs']['total_training_seconds'], fit['costs']['fit_seconds']
                          + fit['costs']['pair_setup_allocation_seconds'], 'complete allocated training time')
        self.exhaust(fits, 'exact six fits')
        expected_prefixes = [{'seed': seed, 'short_final': self.checkpoint_desc['short', seed, 'final'],
                              'long_prefix': self.checkpoint_desc['long', seed, 'prefix'], 'byte_identical': True}
                             for seed in SEEDS]
        self.equal(self.train_summary['prefix_pairs'], expected_prefixes, 'all three saved prefix publication witnesses')
        for key, count in {'initializations': 6, 'final_exports': 6, 'prefix_exports': 3,
                           'parity_numpy_calls': 6, 'parity_torch_calls': 6}.items():
            self.equal(self.train_summary[key], count, 'all fixed training publications and parity calls')
        self.near(self.train_summary['historical_r64_label_seconds'], self.inherited_label_seconds, 'historical cost disclosure')
        self.counts.update(fits=6, epochs=1200, optimizer_updates=6000, training_operations=6060,
                           matched_prefixes=3, training_events=event_count)

    def diagnostics(self):
        np = self.np
        self.context = {'phase': 'saved final TRAIN diagnostics'}
        self.work = iter(self.lines(self.run / 'work.jsonl', maximum=2048,
                                   count_limit=2 * (246 + 1008 + 3 * 504 * H)))
        self.work_counts, self.work_times = collections.Counter(), collections.defaultdict(list)
        records = iter(self.lines(self.run / 'diagnostics.jsonl', maximum=8192, count_limit=6))
        self.diagnostic_summary = self.read(self.run / 'diagnostic-summary.json')
        # Independent geometric D4 mapping: columns are original action IDs;
        # entries are transformed IDs. Reflection of y precedes rotation.
        permutations = [[0, 1, 2, 3], [2, 3, 1, 0], [1, 0, 3, 2], [3, 2, 0, 1],
                        [0, 1, 3, 2], [2, 3, 0, 1], [1, 0, 2, 3], [3, 2, 1, 0]]
        self.equal(self.diagnostic_summary['action_permutations'], permutations, 'qualified D4 action geometry')
        self.equal(self.diagnostic_summary['mse_weights'], 'weights_float32 upcast to float64, sum/N', 'diagnostic MSE weighting')
        self.equal(self.diagnostic_summary['decision_weights'], 'weights_float64, sum/N', 'diagnostic decision weighting')
        self.equal(self.diagnostic_summary['selection'], 'eligible Python-float first within strict 1e-10', 'diagnostic selection rule')

        def operation(name, identity):
            context = {'operation': name, 'index': self.work_counts[name],
                       'phase': 'final_train_diagnostics', **identity}
            self.equal(next(self.work), {'event': 'attempt', **context}, 'diagnostic operation attempt')
            returned = next(self.work)
            self.require(set(returned) == {'event', 'seconds', *context}, 'diagnostic return schema')
            self.require(returned['event'] == 'return' and all(returned[k] == v for k, v in context.items()),
                         'matched diagnostic operation return')
            self.finite(returned['seconds'], 'diagnostic operation interval')
            self.work_counts[name] += 1
            self.work_times[name].append(returned['seconds'])
            return returned['seconds']

        observed = []
        shape_types = {'raw_scores': ((558, 8, 4), np.float32), 'per_view_mse': ((558, 8), np.float64),
            'single_view_mse': ((558,), np.float64), 'eight_view_mse': ((558,), np.float64),
            'selected_action': ((558,), np.int64), 'label_regret': ((558,), np.float64),
            'optimal_set_hit': ((558,), np.bool_), 'first_argmin_match': ((558,), np.bool_)}
        for seed in SEEDS:
            for kind in KINDS:
                self.check()
                fit_id = f'{kind}@{seed}'
                row = next(records)
                self.context = {'phase': 'saved final TRAIN diagnostics', 'fit_id': fit_id}
                self.equal(row['fit_id'], fit_id, 'six fixed diagnostic final checkpoints')
                self.equal(row['checkpoint'], self.checkpoint_desc[kind, seed, 'final'], 'no initial or prefix diagnostics')
                name = f'{kind}-{seed}-diagnostics.npz'
                self.equal(row['file'], {'path': name, **self.worker['files'][name]}, 'diagnostic file descriptor')
                with np.load(self.run / name, allow_pickle=False) as archive:
                    self.require(set(archive.files) == set(shape_types), 'exact diagnostic arrays')
                    arrays = {key: archive[key] for key in archive.files}
                for key, (shape, dtype) in shape_types.items():
                    self.require(arrays[key].shape == shape and arrays[key].dtype == dtype
                                 and np.isfinite(arrays[key]).all(), 'finite typed diagnostic arrays')
                self.equal(row['array_sha256'], {key: hashlib.sha256(value.tobytes()).hexdigest()
                           for key, value in arrays.items()}, 'all diagnostic array hashes')
                restore = operation('diagnostic_restore', {'fit_id': fit_id})
                elapsed = []
                for offset in range(0, 558, 128):
                    for view in range(8):
                        elapsed.append(operation('diagnostic_forward', {'fit_id': fit_id, 'offset': offset,
                                                   'rows': min(128, 558 - offset), 'view': view}))
                self.near(row['restore_seconds'], restore, 'diagnostic restore timing')
                self.near(row['prediction_seconds'], math.fsum(elapsed), 'all forty diagnostic forward intervals')
                self.finite(row['seconds'], 'complete diagnostic interval before journal publication')
                self.require(row['seconds'] >= restore + math.fsum(elapsed), 'diagnostic interval includes restore and predictions')
                self.require(row['forwards'] == 40 and row['view_rows'] == 4464, 'all TRAIN views exactly once')
                target = self.data[f'{kind}_scaled_float32']
                weights32, weights64 = self.data[f'{kind}_weights_float32'], self.data[f'{kind}_weights_float64']
                per_view, decisions, regrets, optimal_hits, first_matches = [], [], [], [], []
                for index in range(558):
                    allowed = np.flatnonzero(self.mask[index]).tolist()
                    mse = []
                    for view, permutation in enumerate(permutations):
                        predicted = [float(arrays['raw_scores'][index, view, permutation[action]]) for action in allowed]
                        targets = [float(target[index, action]) for action in allowed]
                        mean_p, mean_t = math.fsum(predicted) / len(allowed), math.fsum(targets) / len(allowed)
                        mse.append(math.fsum(((p - mean_p) - (t - mean_t)) ** 2
                                             for p, t in zip(predicted, targets, strict=True)) / len(allowed))
                    per_view.append(mse)
                    action = self.saved_choice(arrays['raw_scores'][index, 0].tolist(), allowed)
                    costs = [float(self.data['r64_costs'][index, a]) for a in allowed]
                    best = min(costs)
                    chosen_cost = float(self.data['r64_costs'][index, action])
                    decisions.append(action)
                    regrets.append(chosen_cost - best)
                    optimal_hits.append(chosen_cost == best)
                    first_matches.append(action == allowed[costs.index(best)])
                expected = {'per_view_mse': np.asarray(per_view, dtype=np.float64),
                    'single_view_mse': np.asarray([v[0] for v in per_view], dtype=np.float64),
                    'eight_view_mse': np.asarray([math.fsum(v) / 8 for v in per_view], dtype=np.float64),
                    'selected_action': np.asarray(decisions, dtype=np.int64),
                    'label_regret': np.asarray(regrets, dtype=np.float64),
                    'optimal_set_hit': np.asarray(optimal_hits, dtype=np.bool_),
                    'first_argmin_match': np.asarray(first_matches, dtype=np.bool_)}
                for key, value in expected.items():
                    if key.endswith('_mse'):
                        self.require(np.allclose(arrays[key], value, rtol=1e-12, atol=1e-12),
                                     'independent scalar-fsum diagnostic loss reduction')
                    else:
                        self.require(arrays[key].tobytes() == value.tobytes(), 'exact diagnostic decisions and label regret')
                metrics = {f'{view}_weighted_mse': math.fsum(float(x) * float(w) for x, w in
                           zip(expected[f'{view}_mse'], weights32, strict=True)) / 558 for view in ('single_view', 'eight_view')}
                metrics.update({key: math.fsum(float(x) * float(w) for x, w in zip(expected[key], weights64, strict=True)) / 558
                                for key in ('label_regret', 'optimal_set_hit', 'first_argmin_match')})
                self.require(set(row['metrics']) == set(metrics), 'exact diagnostic metrics')
                for key, value in metrics.items():
                    self.near(row['metrics'][key], value, 'independent episode-weighted TRAIN diagnostic', rtol=1e-12, atol=1e-12)
                observed.append(row)
        self.exhaust(records, 'exact six diagnostic records')
        self.equal(self.diagnostic_summary['fits'], observed, 'diagnostic summary preserves all fit records')
        for key, expected in {'forwards': 240, 'view_rows': 26784, 'checkpoint_restores': 6}.items():
            self.equal(self.diagnostic_summary[key], expected, 'complete fixed diagnostic allocation')
        self.finite(self.diagnostic_summary['seconds'], 'complete TRAIN diagnostic stage')
        self.require(self.diagnostic_summary['seconds'] >= math.fsum(row['seconds'] for row in observed),
                     'stage includes all diagnostic intervals and publication')
        self.counts.update(diagnostic_fits=6, diagnostic_forwards=240, diagnostic_view_rows=26784)

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
        work, counts, times = self.work, self.work_counts, self.work_times
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
                value = means[f'long@{seed}']
                criterion(competence, f'{regime}.{seed}.success', value['found'], .95, value['found'] >= .95)
                criterion(competence, f'{regime}.{seed}.moves', value['steps'], 1.05 * control['steps'],
                          value['steps'] <= 1.05 * control['steps'])
            candidate, baseline = families['long'], families['short']
            gains = [math.fsum(block[f'short@{seed}'] - block[f'long@{seed}'] for seed in SEEDS) / 3
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
                    'inherited_results_revised': False, 'version': 'otto-training-budget-v1'}
        self.tree({key: summary[key] for key in expected}, expected, 'independent study conclusion')
        self.equal(self.worker['pilot_continuation'], passed, 'worker criterion result')
        for value in summary['costs'].values():
            self.finite(value, 'physical stage interval')
        self.near(summary['physical_stage_seconds'], math.fsum(summary['costs'].values()), 'disjoint measured stage subtotal')
        self.require(summary['physical_stage_seconds'] <= self.worker['wall_seconds'] + 1e-6,
                     'stage subtotal fits complete worker time')
        self.near(summary['costs']['train_diagnostic_seconds'], self.diagnostic_summary['seconds'], 'complete diagnostic interval')
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
        self.result = {**expected, 'version': VERSION, 'producer_version': 'otto-training-budget-v1',
                       'criteria_passed': sum(r['passes'] for r in positive + competence + relative_checks),
                       'criteria_total': 33, 'costs': summary['costs'], 'limitations': LIMITATIONS,
                       'training_budget': {'epochs': EPOCHS, 'matched_prefixes': self.prefix_witnesses,
                           'common_original_r64_scale': self.reference_scale,
                           'historical_r64_label_seconds': self.inherited_label_seconds},
                       'train_diagnostics': self.diagnostic_summary,
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
            self.targets()
            self.training()
            self.diagnostics()
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
