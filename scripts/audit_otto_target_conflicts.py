"""Independent saved-only check of exact-input target conflicts.

Reconstructs D4 feature bytes without importing the qualified transform/model,
then checks every saved group, direct weighted deviation and score comparison.
No model, teacher, optimizer, simulator or diagnostic producer is imported.
An external frozen plan and original 120-second supervisor are mandatory.
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
VERSION = 'otto-target-conflicts-saved-audit-v1'
SELF = 'scripts/audit_otto_target_conflicts.py'
PRODUCER = 'scripts/diagnose_otto_target_conflicts.py'
PRODUCER_PIN = '75b37329391e3e8246a49703e255e63f16c6313d22bb30d68509871f3143009d'
PLAN_PIN = 'c0b8cd41d6b72765826cf90c3075e3b27b36d257e44d5f354787062f9e315c43'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
SUPERVISOR_PIN = '610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144'
TRANSFORM = 'src/openjev/research/otto_symmetry_head.py'
TRANSFORM_PIN = 'a18dc9914e4e36d9c8023456dd47919bc7cdbe130cc4fc628c2f3db0044d9b9a'
LIMITS = {'seconds': 120, 'rss_bytes': 1024**3, 'output_bytes': 64 * 1024**2}
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
FITS = tuple(f'{kind}@{seed}' for seed in (30101, 30102, 30103) for kind in ('short', 'long'))
PAYLOADS = {'started.json', 'work.jsonl', 'views.npz', 'memberships.jsonl', 'groups.jsonl', 'comparisons.jsonl', 'summary.json'}
PERMUTATIONS = ((0,1,2,3),(2,3,1,0),(1,0,3,2),(3,2,0,1),
                (0,1,3,2),(2,3,0,1),(1,0,2,3),(3,2,1,0))
LIMITATIONS = [
    'Saved-array arithmetic only; no producer, scientific model, optimizer, sampler or native environment calls.',
    'Historical feature generation, neural score truth and scientific process execution inherit the pinned completed audit.',
    'This verifies a relaxed empirical deterministic-function floor, not a machine-exact Torch loss bound or population noise.',
    'Original timings and durable publication remain authenticated process evidence, not independently re-executed measurements.',
    'All score spreads and signed L-B values are retained; no efficacy threshold or checkpoint promotion is introduced.',
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
        self.require(digest(self.args.audit_plan)['sha256'] == self.args.audit_plan_sha256, 'external audit plan pin')
        plan = decode(self.args.audit_plan.read_bytes())
        self.require(plan['version'] == VERSION and plan['status'] == 'frozen_before_saved_readback'
                     and plan['limits'] == LIMITS and plan['environment'] == THREADS, 'frozen scope and limits')
        sources = {SELF: digest(relative(SELF))['sha256'], CLOCK: CLOCK_PIN, SUPERVISOR: SUPERVISOR_PIN}
        self.equal(plan['sources'], sources, 'exact independent audit sources')
        self.require(set(plan['inputs']) == {'plan', 'worker', 'terminal'}, 'exact external input roles')
        self.require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'one numerical thread')
        for name, pin in sources.items():
            self.require(digest(relative(name))['sha256'] == pin, 'qualified source pin')
        spec = importlib.util.spec_from_file_location('_target_conflicts_saved_clock', relative(CLOCK))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        self.deadline = self.start + 120 * 10**9
        launch = self.read(self.args.supervision)
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and launch['cwd'] == str(ROOT) == str(Path.cwd())
                     and launch['pid'] == os.getpid() == launch['pgid'] != launch['parent_pid'] == os.getppid()
                     and launch['cap_seconds'] == 120 and launch['clock_backend'] == self.clock.backend
                     and launch['clock_source_sha256'] == CLOCK_PIN and launch['watchdog_sha256'] == SUPERVISOR_PIN
                     and launch['deadline_ns'] == launch['started_ns'] + 120 * 10**9
                     and launch['started_ns'] <= self.start < launch['deadline_ns'], 'original audit supervisor')
        self.deadline = launch['deadline_ns']
        self.audit_plan = plan
        self.self_source = self.bind(relative(SELF))
        self.bind(self.args.audit_plan)
        for name in sources:
            self.bind(relative(name))
        self.receipt.update(supervision_sha256=self.bind(self.args.supervision)['sha256'], original_deadline_ns=self.deadline)
        write(self.out / 'started.json', {'audit_plan_sha256': self.args.audit_plan_sha256, 'source': self.self_source,
              'limits': LIMITS, 'inputs': plan['inputs'], 'launch': launch,
              'request': {k: str(v) for k, v in vars(self.args).items()}})

    def authenticate(self):
        entries = self.audit_plan['inputs']
        for role, value in entries.items():
            self.require(self.bind(regular(Path(value['path'])))['sha256'] == value['sha256'], 'external ' + role)
        self.require(entries['plan']['sha256'] == PLAN_PIN, 'fixed diagnostic plan')
        self.plan = self.read(Path(entries['plan']['path']))
        self.worker = self.read(Path(entries['worker']['path']))
        terminal = self.read(Path(entries['terminal']['path']))
        self.run = Path(entries['worker']['path']).parent
        p, w = self.plan, self.worker
        self.receipt['producer_status'] = w.get('status')
        self.require(p['version'] == w['version'] == 'otto-target-conflicts-v1'
                     and p['status'] == 'frozen_before_saved_array_decode' and w['status'] == 'completed'
                     and w['plan_sha256'] == entries['plan']['sha256'], 'closed original diagnostic')
        self.equal(p['limits'], LIMITS, 'original diagnostic bounds')
        self.require(p['runtime']['environment'] == THREADS and p['runtime']['executable'] == sys.executable
                     and p['runtime']['python_version'] == platform.python_version(), 'qualified runtime identity')
        self.require(p['sources'][PRODUCER] == PRODUCER_PIN and p['sources'][TRANSFORM] == TRANSFORM_PIN
                     and p['sources'][CLOCK] == CLOCK_PIN and p['sources'][SUPERVISOR] == SUPERVISOR_PIN,
                     'reviewed producer and qualified dependencies')
        self.require(len(p['sources']) == 6 and len(p['inputs']) == 20, 'exact diagnostic source/input scope')
        engineering = p['engineering']
        self.require(set(engineering) == {f'output/otto-target-conflicts-v1/engineering-01/{name}'
                     for name in ('command-0.log', 'command-1.log', 'receipt.json')}, 'exact fabricated qualification evidence')
        for name, desc in engineering.items():
            self.bind(relative(name), desc)
        for name, pin in p['sources'].items():
            self.require(self.bind(relative(name))['sha256'] == pin, 'frozen diagnostic source')
        for desc in p['inputs'].values():
            self.bind(relative(desc['path']), {k: desc[k] for k in ('bytes', 'sha256')})
        self.equal(w['source'], self.bound[str(relative(PRODUCER))], 'worker producer source descriptor')
        self.equal(w['inputs'], p['inputs'], 'all recorded diagnostic input descriptors')
        self.equal(w['method'], p['method'], 'unchanged diagnostic method')
        self.require(p['method']['rows'] == 558 and p['method']['views'] == 8 and p['method']['occurrences'] == 4464
                     and p['method']['features'] == 2836 and p['method']['arithmetic_atol'] == 1e-12
                     and p['method']['arithmetic_rtol'] == 1e-11, 'fixed all-row diagnostic convention')
        self.require(set(w['files']) == PAYLOADS and {v.name for v in self.run.iterdir()} == PAYLOADS | {'receipt.json'},
                     'complete seven-payload output closure')
        for name, desc in w['files'].items():
            self.bind(self.run / name, desc)
        self.require(w['pending'] == [] and w['scientific_gate'] is None
                     and w['requires_successful_original_supervisor'] is True
                     and all(w[k] == 0 for k in ('model_calls', 'optimizer_calls', 'native_calls', 'sampler_calls')),
                     'closed saved-only work without scientific calls or gate')
        started = self.read(self.run / 'started.json')
        launch = started['launch']
        request = started['request']
        launch_path = regular(Path(request['supervision']))
        self.require(self.bind(launch_path)['sha256'] == w['supervision_sha256'], 'original diagnostic launch pin')
        self.equal(self.read(launch_path), launch, 'embedded original launch')
        self.equal(started['sources'], p['sources'], 'started source binding')
        self.require(all(terminal[k] == v for k, v in launch.items()) and terminal['status'] == 'completed'
                     and terminal['returncode'] == 0 and terminal['timed_out'] is False
                     and terminal['error'] is terminal['clock_error'] is None and terminal['cleanup']['errors'] == []
                     and terminal['group_absent'] is terminal['cleanup']['group_absent'] is terminal['cleanup']['reaped'] is True,
                     'successful original process and cleanup')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(command[:2] == [sys.executable, str(relative(PRODUCER))] and len(command) == 10
                     and len(set(command[2::2])) == 4, 'original diagnostic command')
        self.equal(dict(zip(command[2::2], command[3::2], strict=True)),
                   {'--plan': entries['plan']['path'], '--plan-sha256': entries['plan']['sha256'],
                    '--supervision': str(launch_path), '--output': str(self.run)}, 'original bound diagnostic arguments')
        self.require(request['plan'] == entries['plan']['path'] and request['plan_sha256'] == entries['plan']['sha256']
                     and request['output'] == str(self.run) and launch['cwd'] == str(ROOT)
                     and launch['pid'] == launch['pgid'] != launch['parent_pid'] and launch['cap_seconds'] == 120
                     and launch['clock_backend'] == w['clock_backend'] and launch['clock_source_sha256'] == CLOCK_PIN
                     and launch['watchdog_sha256'] == SUPERVISOR_PIN
                     and launch['deadline_ns'] == launch['started_ns'] + 120 * 10**9
                     and launch['started_ns'] <= w['started_ns'] <= w['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                     'complete original process identity and timing')
        self.near(w['wall_seconds'], (w['finished_ns'] - w['started_ns']) / 1e9, 'worker elapsed arithmetic')
        self.equal(terminal['elapsed_ns'], terminal['finished_ns'] - terminal['started_ns'], 'parent elapsed arithmetic')
        self.near(terminal['wall_seconds'], terminal['elapsed_ns'] / 1e9, 'parent seconds')
        self.require(w['peak_rss_bytes'] <= LIMITS['rss_bytes']
                     and sum(v.stat().st_size for v in self.run.iterdir()) <= LIMITS['output_bytes'], 'original resource bounds')
        prior = self.read(relative(p['inputs']['audit']['path']))
        self.require(p['inputs']['audit']['sha256'] == 'c3ddca8593fb5ffe9f7b01b69728f22f64d80217deb1f38472417fb0c8422b84'
                     and prior['status'] == 'completed' and prior['agreement'] is True, 'inherited completed scientific audit')
        events = iter(self.lines(self.run / 'work.jsonl', maximum=2048, count_limit=28))
        identities = [{'operation': 'transform', 'view': v, 'rows': 558} for v in range(8)] + [
            {'operation': 'score_reduction', 'fit_id': fit, 'rows': 4464} for fit in FITS]
        for identity in identities:
            for event in ('attempt', 'return'):
                self.equal(next(events), {'event': event, **identity}, 'all 28 ordered operation records')
        self.exhaust(events, 'no extra arithmetic operation records')
        self.equal(w['counts'], {'transform_attempted': 8, 'transform_returned': 8,
                                'score_reduction_attempted': 6, 'score_reduction_returned': 6}, 'closed operation counts')
        self.receipt.update(producer_inputs=entries, producer_parent_seconds=terminal['wall_seconds'])

    def transform(self, features, view):
        """Independent grid/coordinate/action reconstruction, without head imports."""
        np = self.np
        value = features.copy()
        grid = features[:, :2809].reshape(558, 53, 53)
        if view >= 4:
            grid = grid[:, :, ::-1]
        value[:, :2809] = np.rot90(grid, view % 4, axes=(1, 2)).reshape(558, 2809)
        x, y = features[:, 2809], features[:, 2810]
        if view >= 4:
            y = -y
        for _ in range(view % 4):
            x, y = -y, x
        value[:, 2809], value[:, 2810] = x, y
        for old, new in enumerate(PERMUTATIONS[view]):
            value[:, 2811 + new] = features[:, 2811 + old]
            value[:, 2816 + 5 * new:2821 + 5 * new] = features[:, 2816 + 5 * old:2821 + 5 * old]
        return value

    def arithmetic(self):
        import numpy as np
        self.np = np
        self.require(np.__version__ == self.plan['runtime']['numpy'], 'qualified NumPy arithmetic version')
        inputs = self.plan['inputs']
        info = self.read(relative(inputs['training_metadata']['path']))
        self.require(len(info['rows']) == 558 and info['episodes'] == 144
                     and [r['anchor_id'] for r in info['rows']] == list(range(558)), 'all ordered TRAIN anchors')
        with np.load(relative(inputs['training_data']['path']), allow_pickle=False) as archive:
            features, target, mask, weights = (archive[k] for k in
                ('features', 'short_scaled_float32', 'allowed', 'short_weights_float32'))
        self.require(features.dtype == target.dtype == weights.dtype == np.float32 and mask.dtype == np.bool_
                     and features.shape == (558, 2836) and target.shape == mask.shape == (558, 4)
                     and weights.shape == (558,) and mask.any(axis=1).all() and (weights > 0).all()
                     and all(np.isfinite(v).all() for v in (features, target, weights)), 'finite typed original cache')
        with np.load(self.run / 'views.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        schemas = {'targets_float32': ((558,8,4), np.float32), 'masks': ((558,8,4), np.bool_),
            'centered_targets_float64': ((558,8,4), np.float64), 'weights_float32': ((558,), np.float32),
            'group_ids': ((558,8), np.int64), 'weighted_deviations': ((558,8), np.float64),
            'fit_per_view_mse': ((6,558,8), np.float64)}
        self.require(set(arrays) == {*schemas, 'fit_group_centered_score_spread'}, 'exact eight saved arrays')
        for key, (shape, dtype) in schemas.items():
            self.require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(),
                         'finite typed saved array')
        self.require(arrays['weights_float32'].tobytes() == weights.tobytes(), 'unchanged f32 weights')
        summary = self.read(self.run / 'summary.json')
        self.equal(summary['arrays'], {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'bytes': v.nbytes,
                   'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}, 'all saved array descriptors')
        self.equal(summary['method'], self.plan['method'], 'reported exact method')
        transformed = [self.transform(features, view) for view in range(8)]
        targets = np.empty((558,8,4), dtype=np.float32)
        masks = np.empty((558,8,4), dtype=np.bool_)
        independent_centered = np.zeros((558,8,4), dtype=np.float64)
        for view, permutation in enumerate(PERMUTATIONS):
            inverse = np.argsort(permutation)
            targets[:, view], masks[:, view] = target[:, inverse], mask[:, inverse]
        self.require(targets.tobytes() == arrays['targets_float32'].tobytes()
                     and masks.tobytes() == arrays['masks'].tobytes(), 'exact independent transformed targets and masks')
        for i in range(558):
            for view in range(8):
                allowed = np.flatnonzero(masks[i, view]).tolist()
                mean = math.fsum(float(targets[i, view, a]) for a in allowed) / len(allowed)
                for a in allowed:
                    independent_centered[i, view, a] = float(targets[i, view, a]) - mean
        # Frozen four-slot arithmetic controls exact zero/nonzero counts.
        # Independently summed means are a comparison, never a changed grouping or threshold.
        target64 = targets.astype(np.float64)
        means = np.where(masks, target64, 0.).sum(axis=-1, keepdims=True, dtype=np.float64) / masks.sum(axis=-1, keepdims=True, dtype=np.int64)
        centered = np.where(masks, target64 - means, 0.)
        self.require(arrays['centered_targets_float64'].tobytes() == centered.tobytes(), 'exact declared four-slot centering bytes')
        self.require(np.allclose(centered, independent_centered, rtol=1e-11, atol=1e-12), 'independent scalar-fsum centering agreement')
        memberships = iter(self.lines(self.run / 'memberships.jsonl', maximum=1024, count_limit=4464))
        raw_groups, groups = {}, []
        for i in range(558):
            self.check()
            for view in range(8):
                xb, mb = transformed[view][i].tobytes(), masks[i, view].tobytes()
                raw = xb + mb
                if raw not in raw_groups:
                    raw_groups[raw] = len(groups)
                    groups.append([])
                group = raw_groups[raw]
                occurrence = i * 8 + view
                groups[group].append(occurrence)
                self.equal(next(memberships), {'occurrence_id': occurrence, 'anchor_id': i, 'view': view, 'group_id': group,
                    'feature_sha256': hashlib.sha256(xb).hexdigest(), 'mask_sha256': hashlib.sha256(mb).hexdigest(),
                    'key_sha256': hashlib.sha256(raw).hexdigest()}, 'full-byte canonical membership')
                self.require(arrays['group_ids'][i, view] == group, 'array group membership')
        self.exhaust(memberships, 'all 4464 memberships without omission')
        group_records = iter(self.lines(self.run / 'groups.jsonl', maximum=128*1024, count_limit=4464))
        flat_y, flat_mask = centered.reshape(-1,4), masks.reshape(-1,4)
        flat_w = np.repeat(weights.astype(np.float64), 8)
        deviations = np.zeros(4464, dtype=np.float64)
        numerators = []
        for gid, members in enumerate(groups):
            self.check()
            saved = next(group_records)
            first = members[0]
            i, view = divmod(first, 8)
            xb, mb = transformed[view][i].tobytes(), flat_mask[first].tobytes()
            allowed = np.flatnonzero(flat_mask[first]).tolist()
            identity = {'group_id': gid, 'members': members, 'representative': first, 'count': len(members),
                'eligible_count': len(allowed), 'mask': flat_mask[first].tolist(),
                'key_sha256': hashlib.sha256(xb+mb).hexdigest(), 'feature_sha256': hashlib.sha256(xb).hexdigest(),
                'mask_sha256': hashlib.sha256(mb).hexdigest()}
            self.equal({k: saved[k] for k in identity}, identity, 'all exact group identities and denominators')
            total = math.fsum(float(flat_w[j]) for j in members)
            sums = [math.fsum(float(flat_w[j])*float(flat_y[j,a]) for j in members) if a in allowed else 0. for a in range(4)]
            mean = [float(flat_y[first,a]) if len(members)==1 else sums[a]/total for a in range(4)]
            for j in members:
                deviations[j] = 0. if len(members)==1 else float(flat_w[j])*math.fsum((float(flat_y[j,a])-mean[a])**2 for a in allowed)/len(allowed)
            numerator = math.fsum(float(deviations[j]) for j in members)
            numerators.append(numerator)
            self.near(saved['weight_sum'], total, 'group summed training weights')
            for key, expected in [('weighted_target_sum',sums),('mean',mean)]:
                self.require(len(saved[key])==4, 'complete action group statistic')
                for value, correct in zip(saved[key],expected,strict=True):
                    self.near(value, correct, 'group weighted target statistic')
            self.near(saved['deviation_numerator'], numerator, 'direct group deviation numerator')
            self.near(saved['floor_contribution'], numerator/4464, 'occurrence-denominated group contribution')
        self.exhaust(group_records, 'no omitted or extra groups')
        self.require(np.allclose(arrays['weighted_deviations'].reshape(-1),deviations,rtol=1e-11,atol=1e-12), 'every direct weighted deviation')
        floor = math.fsum(float(v) for v in deviations)/4464
        stats = {'rows':558,'occurrences':4464,'groups':len(groups),'singleton_groups':sum(len(g)==1 for g in groups),
            'duplicate_groups':sum(len(g)>1 for g in groups),'duplicate_occurrences':sum(len(g) for g in groups if len(g)>1),
            'positive_contribution_groups':sum(v>0 for v in numerators),'maximum_group_size':max(map(len,groups)),
            'denominator':4464,'rounded_weight_sum':math.fsum(float(v) for v in weights)}
        self.equal({k:summary[k] for k in stats},stats,'all group population counts')
        residual = max(abs(math.fsum(float(flat_y[j,a]) for a in range(4) if flat_mask[j,a]) / int(flat_mask[j].sum())) for j in range(4464))
        self.near(summary['maximum_centered_target_mean_residual'],residual,'centering residual disclosed')
        self.near(summary['B'],floor,'common relaxed floor')
        spread_saved = arrays['fit_group_centered_score_spread']
        self.require(spread_saved.shape==(6,len(groups)) and spread_saved.dtype==np.float64
                     and np.isfinite(spread_saved).all(),'complete finite group score spreads')
        prior = self.read(relative(inputs['diagnostic_summary']['path']))
        comparisons = iter(self.lines(self.run/'comparisons.jsonl',maximum=4096,count_limit=6))
        result_fits = []
        for fi, fit in enumerate(FITS):
            self.check()
            self.context = {'phase':'saved-score arithmetic','fit_id':fit}
            record = prior['fits'][fi]
            self.equal(record['fit_id'],fit,'fixed diagnostic fit order')
            with np.load(relative(inputs[fit]['path']),allow_pickle=False) as archive:
                scores = archive['raw_scores']
            self.require(scores.dtype==np.float32 and scores.shape==(558,8,4) and np.isfinite(scores).all(), 'finite saved neural scores')
            self.equal(hashlib.sha256(scores.tobytes()).hexdigest(),record['array_sha256']['raw_scores'],'audited score bytes')
            scores64 = scores.reshape(4464,4).astype(np.float64)
            means = np.where(flat_mask,scores64,0.).sum(axis=-1,keepdims=True,dtype=np.float64) / flat_mask.sum(axis=-1,keepdims=True,dtype=np.int64)
            predictions = np.where(flat_mask,scores64-means,0.)
            losses = np.zeros(4464,dtype=np.float64)
            for j, score in enumerate(scores.reshape(-1,4)):
                allowed = np.flatnonzero(flat_mask[j]).tolist()
                mean = math.fsum(float(score[a]) for a in allowed)/len(allowed)
                for a in allowed:
                    self.near(float(predictions[j,a]),float(score[a])-mean,'independent scalar-fsum score centering')
                losses[j] = math.fsum((float(predictions[j,a])-float(flat_y[j,a]))**2 for a in allowed)/len(allowed)
            self.require(np.allclose(arrays['fit_per_view_mse'][fi].reshape(-1),losses,rtol=1e-11,atol=1e-12),'all independently reduced per-view losses')
            direct = math.fsum(float(flat_w[j])*float(losses[j]) for j in range(4464))/4464
            original = float(np.sum(np.mean(losses.reshape(558,8),axis=1,dtype=np.float64)*weights.astype(np.float64),dtype=np.float64)/558)
            spreads = []
            for members in groups:
                allowed = np.flatnonzero(flat_mask[members[0]]).tolist()
                spreads.append(max(max(float(predictions[j,a]) for j in members)-min(float(predictions[j,a]) for j in members) for a in allowed))
            self.require(np.allclose(spread_saved[fi],spreads,rtol=1e-11,atol=1e-12),'every same-group centered-score spread')
            comparison = next(comparisons)
            identity = {'fit_id':fit,'checkpoint':record['checkpoint'],'scores_sha256':record['array_sha256']['raw_scores'],
                        'groups_with_nonzero_score_spread':sum(v>0 for v in spreads)}
            self.equal({k:comparison[k] for k in identity},identity,'all saved-score identities and nonzero spread counts')
            values = {'L':direct,'B':floor,'L_minus_B':direct-floor,'original_reduction_L':original,
                'reduction_difference':direct-original,'maximum_within_group_centered_score_spread':max(spreads,default=0.),
                'saved_L':record['metrics']['eight_view_weighted_mse'],
                'saved_L_difference':direct-record['metrics']['eight_view_weighted_mse']}
            for key,value in values.items():
                self.near(comparison[key],value,'independent comparison '+key)
            self.near(direct,values['saved_L'],'same previously audited objective')
            result_fits.append(comparison)
        self.exhaust(comparisons,'exact six saved-score comparisons')
        self.equal(summary['fits'],result_fits,'all comparisons retained in summary')
        self.counts.update(memberships=4464,groups=len(groups),array_targets=17856,score_losses=26784,score_group_spreads=6*len(groups),fits=6,operation_events=28)
        self.result = {'version':VERSION,'agreement':True,'summary':summary,'independent_floor':floor,
                       'limitations':LIMITATIONS,'scientific_gate':None}

    def execute(self):
        self.out.mkdir(parents=False,exist_ok=False)
        def interrupted(_signal,_frame):
            raise InterruptedError('saved conflict audit supervisor interrupted')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            self.admission()
            self.authenticate()
            self.arithmetic()
            self.context = {'phase':'closing saved evidence'}
            for name,desc in self.bound.items():
                self.equal(digest(Path(name),self.check),desc,'unchanged evidence at close')
            write(self.out/'audit.json',self.result)
            files = {p.name:digest(p,self.check) for p in self.out.iterdir() if p.is_file()}
            finished = self.clock.now_ns()
            self.receipt.update(status='completed',agreement=True,source=self.self_source,counts=dict(self.counts),
                audit_plan_sha256=self.args.audit_plan_sha256,files=files,started_ns=self.start,finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9,clock_backend=self.clock.backend,scientific_gate=None,
                timing_scope='Through audit payload hashing; original parent covers receipt publication and exit.')
            self.check()
            write(self.out/'receipt.json',self.receipt)
            self.check()
            print(json.dumps({'status':'completed','receipt':digest(self.out/'receipt.json')}),flush=True)
        except BaseException as error:
            self.receipt.update(status='failed',agreement=False,counts=dict(self.counts),
                failures=[{'error':repr(error),'context':self.context,'traceback':traceback.format_exc()}])
            try:
                destination=self.out/'receipt.json'
                if destination.exists():
                    destination.rename(self.out/'receipt.invalid.json')
                self.receipt['partial_files']={p.name:{'bytes':p.stat().st_size} for p in self.out.iterdir() if p.is_file()}
                write(destination,self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original terminal failure
                error.add_note(f'Failure receipt publication: {secondary!r}')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-plan',type=Path,required=True)
    parser.add_argument('--audit-plan-sha256',required=True)
    parser.add_argument('--supervision',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('audit_plan','supervision','output'):
        regular(getattr(args,name))
    Audit(args).execute()


if __name__=='__main__':
    main()
