"""Saved-only exact-input conflict floor; no learning, prediction or admission gate."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
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
SELF = 'scripts/diagnose_otto_target_conflicts.py'
TEST = 'tests/test_otto_target_conflicts.py'
PROTOCOL = 'research/otto-target-conflicts-protocol.md'
VERSION = 'otto-target-conflicts-v1'
BASE = 'output/otto-training-budget-v1/'
CLOCK = 'src/openjev/research/suspend_clock.py'
TRANSFORM = 'src/openjev/research/otto_symmetry_head.py'
SUPERVISOR = 'scripts/supervise_dialogue_observation_v2.py'
FIXED_SOURCES = {
    CLOCK: 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124',
    TRANSFORM: 'a18dc9914e4e36d9c8023456dd47919bc7cdbe130cc4fc628c2f3db0044d9b9a',
    SUPERVISOR: '610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144',
}
PINS = {
    'empirical_plan': (BASE + 'plan-01.json', 'a204495fb0fd66ea3e7ea7c5052315f85bf3fa75438c243cf6a6f17e92e4099a'),
    'worker': (BASE + 'run-01/receipt.json', '47004eb796ac7ab9a836f88c2942dd6b6f00e008d277b205afbdf85e3a8eb182'),
    'worker_parent': (BASE + 'supervision-01.terminal.json', 'd99dbbd53bb3ca281b15db82f577f5e8ea6e62ddecb210a40b86e73004b2d276'),
    'audit': (BASE + 'audit-01/receipt.json', 'c3ddca8593fb5ffe9f7b01b69728f22f64d80217deb1f38472417fb0c8422b84'),
    'audit_plan': (BASE + 'audit-plan-01.json', 'c0660c831d254b1334448663f3ace3f6c8809935f8a4aeb0b1601ab9a5173714'),
    'audit_parent': (BASE + 'audit-supervision-01.terminal.json', '1bea061a7463349c0787896300d569453b09eb1fdbd4a1fb96102beab4f68bb9'),
}
FITS = tuple(f'{kind}@{seed}' for seed in (30101, 30102, 30103) for kind in ('short', 'long'))
LIMITS = {'seconds': 120, 'rss_bytes': 1024**3, 'output_bytes': 64 * 1024**2}
THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')
METHOD = {
    'rows': 558, 'views': 8, 'occurrences': 4464, 'features': 2836,
    'grouping': 'Full transformed float32 feature C-bytes plus bool mask C-bytes; SHA buckets with full-byte collision verification. Signed zeros retained.',
    'targets': 'Saved rounded float32 targets and weights upcast to float64; eligible four-slot mean subtraction.',
    'floor': 'Weighted free group means; direct weighted eligible mean squared deviations; denominator N*8, not weight sum. Singleton contribution exactly zero.',
    'score_spread': 'Maximum eligible-coordinate max-minus-min of centered saved float32 scores upcast to float64 within each exact group; singleton zero.',
    'comparison': 'Six full saved-score MSEs L, common B and signed L-B; no clipping, selection or pass gate.',
    'arithmetic_atol': 1e-12, 'arithmetic_rtol': 1e-11,
    'scope': 'Relaxed empirical real deterministic-function floor, not a machine-exact Torch loss bound, population noise estimate or control efficacy result.',
}
PAYLOADS = {'started.json', 'work.jsonl', 'views.npz', 'memberships.jsonl', 'groups.jsonl',
            'comparisons.jsonl', 'summary.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def bad(value):
        raise ValueError('nonfinite JSON ' + value)
    require(path.stat().st_size <= 8 * 1024**2, 'bounded metadata')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=bad)


def digest(path, check=lambda: None):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink file')
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            h.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    return {'executable': sys.executable, 'python_version': platform.python_version(),
            'numpy': importlib.metadata.version('numpy'), 'environment': THREADS}


def source_names():
    return {SELF, TEST, PROTOCOL, *FIXED_SOURCES}


def input_descriptors(check=lambda: None):
    """Plan assembly helper: hash only metadata and declared current payloads."""
    result = {}

    def add(role, name, expected=None, pin=None):
        value = digest(ROOT / name, check)
        require(expected is None or value == expected, 'payload descriptor ' + role)
        require(pin is None or value['sha256'] == pin, 'external lineage pin ' + role)
        result[role] = {'path': name, **value}

    for role, (name, pin) in PINS.items():
        add(role, name, pin=pin)
    worker, audit = (read(ROOT / PINS[role][0]) for role in ('worker', 'audit'))
    roles = {'training_data': 'training-data.npz', 'training_metadata': 'training-data.json',
             'diagnostic_summary': 'diagnostic-summary.json', 'worker_started': 'started.json'}
    roles.update({fit: fit.replace('@', '-') + '-diagnostics.npz' for fit in FITS})
    for role, name in roles.items():
        add(role, BASE + 'run-01/' + name, worker['files'][name])
    for role, name in (('audit_summary', 'audit.json'), ('audit_started', 'started.json')):
        add(role, BASE + 'audit-01/' + name, audit['files'][name])
    for role, prefix, receipt in (('worker_launch', 'supervision-01', worker), ('audit_launch', 'audit-supervision-01', audit)):
        add(role, BASE + prefix + '.launch.json', pin=receipt['supervision_sha256'])
    return result


def key_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def centered(values, masks):
    import numpy as np
    value = values.astype(np.float64)
    count = masks.sum(axis=-1, keepdims=True, dtype=np.int64)
    require((count > 0).all(), 'nonempty eligible mask')
    mean = np.where(masks, value, 0.).sum(axis=-1, keepdims=True, dtype=np.float64) / count
    return np.where(masks, value - mean, 0.)


def compute_conflicts(features, targets, masks, weights, *, transform_features, transform_actions,
                      check=lambda: None, observe=lambda _event: None):
    """Pure all-row calculation; supplied transforms are the pinned D4 functions."""
    import numpy as np
    n = len(features)
    require(n > 0 and features.dtype == targets.dtype == weights.dtype == np.float32
            and features.shape == (n, 2836) and targets.shape == masks.shape == (n, 4)
            and masks.dtype == np.bool_ and weights.shape == (n,) and masks.any(axis=1).all()
            and all(np.isfinite(v).all() for v in (features, targets, weights)) and (weights > 0).all(),
            'finite exact f32 inputs and positive weights')
    xs, ys, ms = [], [], []
    for view in range(8):
        check()
        observe({'event': 'attempt', 'operation': 'transform', 'view': view, 'rows': n})
        x, y, m = transform_features(features, view), transform_actions(targets, view), transform_actions(masks, view)
        require(x.shape == features.shape and x.dtype == np.float32 and y.shape == targets.shape
                and y.dtype == np.float32 and m.shape == masks.shape and m.dtype == np.bool_, 'exact transformed schema')
        xs.append(x)
        ys.append(y)
        ms.append(m)
        observe({'event': 'return', 'operation': 'transform', 'view': view, 'rows': n})
    target, mask = np.stack(ys, axis=1), np.stack(ms, axis=1)
    y = centered(target, mask)
    buckets, full_keys, groups, memberships = {}, [], [], []
    ids = np.empty((n, 8), dtype=np.int64)
    for row in range(n):
        check()
        for view in range(8):
            feature_bytes, mask_bytes = xs[view][row].tobytes(order='C'), mask[row, view].tobytes(order='C')
            raw = feature_bytes + mask_bytes
            hashed = key_hash(raw)
            group_id = next((g for g in buckets.get(hashed, ()) if full_keys[g] == raw), None)
            if group_id is None:
                group_id = len(groups)
                full_keys.append(raw)
                buckets.setdefault(hashed, []).append(group_id)
                groups.append({'group_id': group_id, 'members': [], 'key_sha256': hashlib.sha256(raw).hexdigest(),
                               'feature_sha256': hashlib.sha256(feature_bytes).hexdigest(),
                               'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(), 'mask': mask[row, view].tolist()})
            occurrence = row * 8 + view
            groups[group_id]['members'].append(occurrence)
            ids[row, view] = group_id
            memberships.append({'occurrence_id': occurrence, 'anchor_id': row, 'view': view, 'group_id': group_id,
                                'feature_sha256': hashlib.sha256(feature_bytes).hexdigest(),
                                'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(), 'key_sha256': hashlib.sha256(raw).hexdigest()})
    deviations = np.zeros((n, 8), dtype=np.float64)
    flat_y = y.reshape(-1, 4)
    flat_w = np.repeat(weights.astype(np.float64), 8)
    for group in groups:
        check()
        members = group['members']
        allowed = [a for a, flag in enumerate(group['mask']) if flag]
        total = math.fsum(float(flat_w[j]) for j in members)
        sums = [math.fsum(float(flat_w[j]) * float(flat_y[j, a]) for j in members) if a in allowed else 0.
                for a in range(4)]
        mean = [float(flat_y[members[0], a]) if len(members) == 1 else sums[a] / total for a in range(4)]
        for j in members:
            deviations.flat[j] = (0. if len(members) == 1 else float(flat_w[j]) *
                math.fsum((float(flat_y[j, a]) - mean[a])**2 for a in allowed) / len(allowed))
        numerator = math.fsum(float(deviations.flat[j]) for j in members)
        group.update(representative=members[0], count=len(members), eligible_count=len(allowed), weight_sum=total,
                     weighted_target_sum=sums, mean=mean, deviation_numerator=numerator,
                     floor_contribution=numerator / (n * 8))
    arrays = {'targets_float32': target, 'masks': mask, 'centered_targets_float64': y, 'weights_float32': weights.copy(),
              'group_ids': ids, 'weighted_deviations': deviations}
    floor = math.fsum(float(v) for v in deviations.flat) / (n * 8)
    summary = {'rows': n, 'occurrences': n * 8, 'groups': len(groups),
               'singleton_groups': sum(g['count'] == 1 for g in groups),
               'duplicate_groups': sum(g['count'] > 1 for g in groups),
               'duplicate_occurrences': sum(g['count'] for g in groups if g['count'] > 1),
               'positive_contribution_groups': sum(g['deviation_numerator'] > 0 for g in groups),
               'maximum_group_size': max(g['count'] for g in groups), 'denominator': n * 8,
               'rounded_weight_sum': math.fsum(float(v) for v in weights), 'B': floor,
               'maximum_centered_target_mean_residual': max(abs(math.fsum(float(y[i, v, a]) for a in range(4)
                   if mask[i, v, a]) / int(mask[i, v].sum())) for i in range(n) for v in range(8))}
    require(math.isfinite(floor), 'finite floor')
    return arrays, memberships, groups, summary


def compare_scores(scores, arrays, groups, floor):
    """Reduce existing saved scores only; signed excess and spreads are descriptive."""
    import numpy as np
    masks, targets = arrays['masks'], arrays['centered_targets_float64']
    n = len(targets)
    require(scores.dtype == np.float32 and scores.shape == (n, 8, 4) and np.isfinite(scores).all(), 'saved raw scores')
    predicted = centered(scores, masks)
    error = np.where(masks, predicted - targets, 0.)
    losses = np.sum(error * error, axis=-1, dtype=np.float64) / masks.sum(axis=-1, dtype=np.int64)
    weights = arrays['weights_float32'].astype(np.float64)
    direct = math.fsum(float(weights[i]) * float(losses[i, v]) for i in range(n) for v in range(8)) / (n * 8)
    original_route = float(np.sum(np.mean(losses, axis=1, dtype=np.float64) * weights, dtype=np.float64) / n)
    flat = predicted.reshape(-1, 4)
    spread = np.zeros(len(groups), dtype=np.float64)
    for group in groups:
        if group['count'] > 1:
            values = flat[group['members']][:, group['mask']]
            spread[group['group_id']] = float(np.max(values.max(axis=0) - values.min(axis=0)))
    require(np.isfinite(losses).all() and np.isfinite(spread).all() and math.isfinite(direct), 'finite comparison')
    result = {'L': direct, 'B': floor, 'L_minus_B': direct - floor, 'original_reduction_L': original_route,
              'reduction_difference': direct - original_route,
              'maximum_within_group_centered_score_spread': float(spread.max(initial=0.)),
              'groups_with_nonzero_score_spread': int(np.count_nonzero(spread))}
    return result, losses, spread


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock, self.start, self.deadline = None, None, None
        self.bound, self.pending, self.counts = {}, {}, collections.Counter()
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS,
                        'model_calls': 0, 'optimizer_calls': 0, 'native_calls': 0, 'sampler_calls': 0}

    def check(self):
        if self.clock is not None:
            require(self.clock.now_ns() < self.deadline, 'original 120-second deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
                <= LIMITS['rss_bytes'], 'RSS limit')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'output limit')

    def bind(self, path, expected=None, pin=None):
        actual = digest(path, self.check)
        require(expected is None or actual == expected, 'descriptor identity ' + str(path))
        require(pin is None or actual['sha256'] == pin, 'pinned identity ' + str(path))
        self.bound[str(path)] = actual
        return actual

    def event(self, value):
        key = (value['operation'], value.get('view', value.get('fit_id')))
        if value['event'] == 'attempt':
            require(key not in self.pending, 'unique pending operation')
            self.pending[key] = value
            self.counts[value['operation'] + '_attempted'] += 1
        else:
            require(key in self.pending and {**self.pending[key], 'event': 'return'} == value,
                    'returned matching pending operation')
        with (self.out / 'work.jsonl').open('ab') as stream:
            raw = encoded(value)
            require(stream.write(raw) == len(raw), 'complete operation event')
            stream.flush()
            os.fsync(stream.fileno())
        if value['event'] == 'return':
            self.counts[value['operation'] + '_returned'] += 1
            del self.pending[key]

    def admit(self):
        require(all(os.environ.get(k) == v for k, v in THREADS.items()), 'single numerical thread')
        self.bind(self.args.plan, pin=self.args.plan_sha256)
        self.plan = read(self.args.plan)
        plan = self.plan
        require(plan['version'] == VERSION and plan['status'] == 'frozen_before_saved_array_decode'
                and plan['method'] == METHOD and plan['limits'] == LIMITS and plan['runtime'] == runtime(), 'frozen diagnostic contract')
        require(set(plan['sources']) == source_names(), 'exact diagnostic sources')
        for name, pin in plan['sources'].items():
            require(name not in FIXED_SOURCES or pin == FIXED_SOURCES[name], 'qualified dependency pin')
            self.bind(ROOT / name, pin=pin)
        spec = importlib.util.spec_from_file_location('_target_conflicts_clock', ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        launch = read(self.args.supervision)
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch['cwd'] == str(ROOT) == str(Path.cwd())
                and launch['pid'] == os.getpid() == launch['pgid'] != launch['parent_pid'] == os.getppid()
                and launch['cap_seconds'] == 120 and launch['clock_backend'] == self.clock.backend
                and launch['clock_source_sha256'] == FIXED_SOURCES[CLOCK]
                and launch['watchdog_sha256'] == FIXED_SOURCES[SUPERVISOR]
                and launch['deadline_ns'] == launch['started_ns'] + 120 * 10**9
                and launch['started_ns'] <= self.start < launch['deadline_ns'], 'original diagnostic supervisor')
        self.deadline = launch['deadline_ns']
        self.bind(self.args.supervision)
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=self.bound[str(self.args.supervision)]['sha256'],
                            started_ns=self.start, clock_backend=self.clock.backend, requires_successful_original_supervisor=True)
        write(self.out / 'started.json', {**self.receipt, 'launch': launch, 'runtime': runtime(),
              'request': {k: str(v) for k, v in vars(self.args).items()}, 'sources': plan['sources']})
        expected = input_descriptors(self.check)
        require(plan['inputs'] == expected, 'all required current input descriptors')
        for value in expected.values():
            self.bind(ROOT / value['path'], {k: value[k] for k in ('sha256', 'bytes')})
        get = lambda role: read(ROOT / expected[role]['path'])
        parent, worker, audit = get('empirical_plan'), get('worker'), get('audit')
        require(parent['status'] == 'frozen_before_training' and worker['status'] == audit['status'] == 'completed'
                and worker['version'] == 'otto-training-budget-v1' and audit['version'] == 'otto-training-budget-saved-audit-v1'
                and audit['agreement'] is True and worker['plan_sha256'] == PINS['empirical_plan'][1]
                and audit['audit_plan_sha256'] == PINS['audit_plan'][1] and worker['pending'] == []
                and worker['pending_episode'] is None and worker['completed_diagnostics'] == worker['completed_fits'] == 6
                and worker['completed_episodes'] == 504 and len(worker['files']) == 34, 'complete audited producer')
        require(audit['producer_inputs'] == {role: {'path': str(ROOT / PINS[key][0]), 'sha256': PINS[key][1]}
                for role, key in (('plan', 'empirical_plan'), ('worker', 'worker'), ('terminal', 'worker_parent'))}
                and get('audit_plan')['inputs'] == audit['producer_inputs'], 'independent audit identity joins')
        for prefix, receipt in (('worker', worker), ('audit', audit)):
            launch, terminal, started = get(prefix + '_launch'), get(prefix + '_parent'), get(prefix + '_started')
            require(started['launch'] == launch and all(terminal[k] == v for k, v in launch.items())
                    and terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                    and terminal['error'] is terminal['clock_error'] is None and terminal['cleanup']['errors'] == []
                    and terminal['group_absent'] is terminal['cleanup']['group_absent'] is terminal['cleanup']['reaped'] is True
                    and launch['pid'] == launch['pgid'] != launch['parent_pid'] and launch['cwd'] == str(ROOT)
                    and launch['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns']
                    <= terminal['finished_ns'] < launch['deadline_ns'], 'original completed process ' + prefix)
        require(len(parent['sources']) == 164 and len(parent['inputs']) == 1748, 'inherited manifest counts')
        for name, pin in parent['sources'].items():
            self.bind(ROOT / name, pin=pin)
        self.inputs, self.worker, self.parent, self.checked = expected, worker, parent, get('audit_summary')

    def compute(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location('_target_conflicts_transform', ROOT / TRANSFORM)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        metadata = read(ROOT / self.inputs['training_metadata']['path'])
        require(metadata['rows'] == self.parent['selections'] and len(metadata['rows']) == 558
                and metadata['episodes'] == 144, 'entire fixed TRAIN metadata')
        fields = ('centered_float64', 'scaled_float32', 'weights_float64', 'weights_float32', 'allowed')
        with np.load(ROOT / self.inputs['training_data']['path'], allow_pickle=False) as archive:
            require(set(archive.files) == {'features', 'r64_costs', 'allowed', 'anchor_ids'} |
                    {f'{kind}_{key}' for kind in ('short', 'long') for key in fields}, 'exact cache schema')
            features, masks = archive['features'], archive['allowed']
            require(np.array_equal(archive['anchor_ids'], np.arange(558, dtype=np.int64)), 'canonical anchor order')
            target, weights = archive['short_scaled_float32'], archive['short_weights_float32']
            for key, value in (('scaled_float32', target), ('weights_float32', weights), ('allowed', masks)):
                for kind in ('short', 'long'):
                    saved = archive[f'{kind}_{key}']
                    require(saved.dtype == value.dtype and saved.shape == value.shape
                            and saved.tobytes() == value.tobytes(), 'unchanged shared input bytes')
        counts = collections.Counter(row['episode_id'] for row in metadata['rows'])
        require(len(counts) == 144 and hashlib.sha256(features.tobytes()).hexdigest() == metadata['features_sha256']
                and masks.tolist() == [[a in row['public']['valid_actions'] for a in range(4)] for row in metadata['rows']]
                and weights.tobytes() == np.array([558 / (144 * counts[row['episode_id']]) for row in metadata['rows']],
                                                dtype=np.float32).tobytes(), 'fixed features, eligibility and episode weights')
        arrays, memberships, groups, summary = compute_conflicts(features, target, masks, weights,
            transform_features=module.transform_features, transform_actions=module.transform_actions,
            check=self.check, observe=self.event)
        diagnostic = read(ROOT / self.inputs['diagnostic_summary']['path'])
        require(diagnostic == self.checked['train_diagnostics'] and [r['fit_id'] for r in diagnostic['fits']] == list(FITS),
                'all six independently audited diagnostic identities')
        comparisons, losses, spreads = [], [], []
        for fit, record in zip(FITS, diagnostic['fits'], strict=True):
            self.check()
            self.event({'event': 'attempt', 'operation': 'score_reduction', 'fit_id': fit, 'rows': 558 * 8})
            desc = self.inputs[fit]
            require(record['file'] == {'path': Path(desc['path']).name, 'sha256': desc['sha256'], 'bytes': desc['bytes']},
                    'diagnostic NPZ record join')
            with np.load(ROOT / desc['path'], allow_pickle=False) as archive:
                scores = archive['raw_scores']
            require(hashlib.sha256(scores.tobytes()).hexdigest() == record['array_sha256']['raw_scores'], 'saved score bytes')
            result, loss, spread = compare_scores(scores, arrays, groups, summary['B'])
            saved_loss = record['metrics']['eight_view_weighted_mse']
            require(math.isclose(result['L'], saved_loss, rel_tol=METHOD['arithmetic_rtol'], abs_tol=METHOD['arithmetic_atol'])
                    and math.isclose(result['original_reduction_L'], saved_loss,
                                     rel_tol=METHOD['arithmetic_rtol'], abs_tol=METHOD['arithmetic_atol']), 'same saved objective')
            comparisons.append({'fit_id': fit, **result, 'saved_L': saved_loss, 'saved_L_difference': result['L'] - saved_loss,
                                'checkpoint': record['checkpoint'], 'scores_sha256': record['array_sha256']['raw_scores']})
            losses.append(loss)
            spreads.append(spread)
            self.event({'event': 'return', 'operation': 'score_reduction', 'fit_id': fit, 'rows': 558 * 8})
        arrays['fit_per_view_mse'] = np.stack(losses)
        arrays['fit_group_centered_score_spread'] = np.stack(spreads)
        with (self.out / 'views.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        for name, rows in (('memberships', memberships), ('groups', groups), ('comparisons', comparisons)):
            with (self.out / f'{name}.jsonl').open('xb') as stream:
                for row in rows:
                    self.check()
                    stream.write(encoded(row))
                stream.flush()
                os.fsync(stream.fileno())
        write(self.out / 'summary.json', {'version': VERSION, 'method': METHOD, **summary, 'fits': comparisons,
              'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'bytes': v.nbytes,
                             'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()},
              'provenance_scope': 'Current required arrays and metadata plus all 164 scientific sources rehashed; other payloads and ancestral input closure inherited from pinned completed audit. No panel, trajectory, model or optimizer replay.',
              'score_scope': 'Saved predictions may differ slightly within identical-input groups due to floating-point batch context; spreads are reported without correcting L or B.'})

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupted(_signal, _frame):
            raise InterruptedError('original diagnostic supervision interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit()
            self.compute()
            require(not self.pending and self.counts == {'transform_attempted': 8, 'transform_returned': 8,
                    'score_reduction_attempted': 6, 'score_reduction_returned': 6}, 'all fixed arithmetic operations complete')
            for name, desc in self.bound.items():
                require(digest(Path(name), self.check) == desc, 'unchanged bound evidence')
            require({p.name for p in self.out.iterdir()} == PAYLOADS, 'exact completed payload closure')
            files = {name: digest(self.out / name, self.check) for name in sorted(PAYLOADS)}
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', files=files, counts=dict(self.counts), pending=[],
                source=self.bound[str(ROOT / SELF)], started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished - self.start) / 1e9,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                inputs=self.inputs, method=METHOD, scientific_gate=None,
                timing_scope='Worker through final input/source/payload hashing; original parent covers receipt publication and exit.')
            self.check()
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'receipt': digest(self.out / 'receipt.json')}), flush=True)
        except BaseException as error:
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                                pending=list(self.pending.values()), counts=dict(self.counts))
            try:
                path = self.out / 'receipt.json'
                if path.exists():
                    path.rename(self.out / 'receipt.invalid.json')
                self.receipt['partial_files'] = {p.name: {'bytes': p.stat().st_size} for p in self.out.iterdir() if p.is_file()}
                write(path, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain the primary failure
                error.add_note(f'Failure receipt publication: {secondary!r}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--supervision', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(all(getattr(args, key).is_absolute() for key in ('plan', 'supervision', 'output')), 'absolute paths required')
    Run(args).execute()


if __name__ == '__main__':
    main()
