"""Independent saved-only return-value audit; no training or simulator calls.

Qualified prior authentication/filter/journal helpers are inherited. Scalar
features, targets, exported networks, sixteen branches, choices and aggregate
decisions are reconstructed here without producer numerical functions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/audit_otto_symmetry_head.py'
HELPER_PIN = 'd17b1269529b969d45d4f3d0079b27a8d44caac2212c2cea4a0230e5515a94b0'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('frozen independent audit helper identity')
_spec = importlib.util.spec_from_file_location('_return_value_independent_helpers', HELPER)
A = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = A
_spec.loader.exec_module(A)
B, require, Work = A.B, A.require, A.Work

VERSION = 'otto-return-value-saved-audit-v1'
RUNNER = 'scripts/study_otto_return_value.py'
LIMITS = {'native_seconds': 1800, 'rss_bytes': 4 * 1024**3, 'output_bytes': 128 * 1024**2}
DIMENSION, HORIZON = 11028, 2188
SEEDS = (10101, 10102, 10103)
FAMILIES = ('min8', 'mlp8', 'homogeneous8')
ARMS = tuple(f'{f}@{s}' for s in SEEDS for f in FAMILIES) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'lambda3': 11100001, 'lambda4': 11200001, 'lambda5': 11300001}
TIMES = ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')
METRICS = ('steps', 'found', *TIMES, 'controller_seconds', 'environment_seconds', 'state_bytes')
SCOPE = ('Independent all-prefix TRAIN/VALID filtering, scalar targets and TRAIN-only baseline; '
         'complete exported checkpoints, saved scalar and sixteen-branch NumPy readouts, public '
         'evaluation updates/actions, recorded work/cost accounting and54 scientific conditions. '
         'No training, simulator or remote calls. Training trajectories, native RNG/runtime, '
         'original teacher behavior, Torch outputs and timing truth remain producer evidence. '
         'Local saved-checkpoint readouts are explicitly counted.')


def features(centered, positions, sensing_length, np, *, dtype='float64'):
    require(centered.ndim == 3 and centered.shape[1:] == (105, 105)
            and centered.dtype == np.float64 and np.isfinite(centered).all()
            and (centered >= 0).all(), 'finite nonnegative centered float64 fields')
    require(positions.shape == (len(centered), 2) and positions.dtype.kind in 'iu'
            and (positions >= 0).all() and (positions <= 52).all()
            and sensing_length in (3., 4., 5.) and dtype in ('float32', 'float64'), 'public context geometry')
    flat = centered.reshape(len(centered), 11025)
    mass = flat.sum(axis=1, dtype=np.float64)
    context = np.column_stack((mass * positions[:, 0] / 52,
                               mass * positions[:, 1] / 52, mass * sensing_length / 5))
    return np.concatenate((flat, context), axis=1).astype(dtype)


def state_features(probability, position, sensing_length, np):
    x, y = position
    field = np.zeros((1, 105, 105), np.float64)
    field[0, 52-x:105-x, 52-y:105-y] = probability
    return features(field, np.asarray([position], np.int64), sensing_length, np, dtype='float32')[0]


def targets(steps, np):
    require(type(steps) is int and 1 <= steps <= HORIZON, 'complete positive episode duration')
    return np.asarray([(steps-t)/64 for t in range(steps)], dtype=np.float32)


def tensor_names(kind):
    require(kind in FAMILIES, 'declared scalar architecture')
    return {'first_weight'} | ({'final_weight'} if kind != 'min8' else set()) | (
        {'hidden_bias', 'output_bias'} if kind == 'mlp8' else set())


def checkpoint(archive, kind, c0, np):
    require(set(archive) == {'version', 'kind', 'input_dim', 'c0', *tensor_names(kind)}, 'exact scalar checkpoint closure')
    require(archive['version'].item() == 'otto-return-value-v1' and archive['kind'].item() == kind
            and archive['input_dim'].item() == DIMENSION, 'scalar checkpoint identity')
    shapes = {'first_weight': (8, DIMENSION), 'final_weight': (8,), 'hidden_bias': (8,), 'output_bias': ()}
    for key in {'c0', *tensor_names(kind)}:
        value = archive[key]
        require(value.dtype == np.float32 and value.shape == (() if key == 'c0' else shapes[key])
                and np.isfinite(value).all(), 'exact finite float32 scalar checkpoint tensors')
    require(float(archive['c0']) == c0, 'same TRAIN-only baseline in every checkpoint')
    return {key: archive[key].astype(np.float64) for key in {'c0', *tensor_names(kind)}}


def predict(x, weights, kind, np):
    require(x.ndim == 2 and x.shape[1] == DIMENSION and x.dtype == np.float64
            and np.isfinite(x).all(), 'float64 scalar readout inputs')
    first = x @ weights['first_weight'].T
    if kind == 'min8':
        residual = first.min(axis=1)
    else:
        hidden = first + weights['hidden_bias'] if kind == 'mlp8' else first
        residual = np.maximum(hidden, 0) @ weights['final_weight']
        if kind == 'mlp8':
            residual = residual + weights['output_bias']
    result = weights['c0'] * x[:, :11025].sum(axis=1, dtype=np.float64) + residual
    require(result.shape == (len(x),) and np.isfinite(result).all(), 'finite signed scalar predictions')
    return result


def branches(probability, position, kernel, sensing_length, np):
    """Independent action-major/hit-minor crop, pre-centering mass and floor."""
    fields = np.zeros((16, 105, 105), np.float64)
    positions, masses = [], []
    for action in range(4):
        x, y = B.move(position, action)
        for hit in range(4):
            joint = probability * kernel[hit, 53-x:106-x, 53-y:106-y]
            mass = float(joint.sum(dtype=np.float64))
            masses.append(mass)
            positions.append([x, y])
            fields[4*action+hit, 52-x:105-x, 52-y:105-y] = joint / max(mass, 1e-10)
    raw = np.asarray(masses, np.float64).reshape(4, 4)
    return features(fields, np.asarray(positions, np.int64), sensing_length, np), raw, np.maximum(raw, 1e-10)


def costs(values, weights, np):
    require(values.shape == (16,) and values.dtype == np.float64 and np.isfinite(values).all(), 'all sixteen finite values')
    return 1 + (weights * values.reshape(4, 4)).sum(axis=1, dtype=np.float64)


def choice(recorded, allowed, learned, np):
    require(len(recorded) == 4 and allowed == sorted(set(allowed)) and allowed
            and all(type(a) is int and 0 <= a < 4 for a in allowed), 'four costs and ordered public eligibility')
    require(all((type(v) in (int, float) and math.isfinite(v)) if learned or i in allowed else v is None
                for i, v in enumerate(recorded)), 'finite learned raw costs or analytic null mask')
    value = np.asarray([math.inf if v is None else v for v in recorded], np.float64)
    legal = value[allowed]
    return allowed[int(np.flatnonzero(np.abs(legal-legal.min()) < 1e-10)[0])]


def evaluation_order():
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri*24+case) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield regime, first+case, 1+case % 3, arm, case//3


def condition(name, value, threshold, passed):
    return {'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passed)}


def aggregate(rows, mixtures):
    require([(r['regime'], r['seed'], r['initial_hit'], r['arm'], r['block']) for r in rows]
            == list(evaluation_order()), 'all720 exact evaluation identities and rotation')
    regimes, competence, utility, architecture = {}, [], [], []
    for regime in REGIMES:
        weight = {int(h): float(w) for h, w in mixtures[regime].items()}
        require(set(weight) == {1, 2, 3} and all(math.isfinite(w) and 0 < w < 1 for w in weight.values())
                and abs(math.fsum(weight.values())-1) <= 1e-12, 'complete positive initial mixture')
        subset = [r for r in rows if r['regime'] == regime]

        def mean(selected, metric, weight=weight):
            parts = [[float(r[metric]) for r in selected if r['initial_hit'] == h] for h in (1, 2, 3)]
            require(all(parts), 'all three conditional strata')
            return math.fsum(weight[h] * math.fsum(v)/len(v) for h, v in zip((1, 2, 3), parts, strict=True))

        means = {arm: {m: mean([r for r in subset if r['arm'] == arm], m) for m in METRICS} for arm in ARMS}
        blocks = [{arm: mean([r for r in subset if r['arm'] == arm and r['block'] == b], 'steps') for arm in ARMS} for b in range(8)]
        families = {f: {m: math.fsum(means[f'{f}@{s}'][m] for s in SEEDS)/3 for m in METRICS} for f in FAMILIES}
        candidate, reference = families['min8'], means['analytic_inbounds']
        for seed in SEEDS:
            fit = means[f'min8@{seed}']
            competence.extend([condition(f'{regime}.{seed}.success', fit['found'], .95, fit['found'] >= .95),
                               condition(f'{regime}.{seed}.moves', fit['steps'], 1.05*reference['steps'], fit['steps'] <= 1.05*reference['steps'])])
        worst = max(means[f'min8@{s}']['controller_seconds'] for s in SEEDS)
        utility.extend([condition(f'{regime}.success', candidate['found'], reference['found'], candidate['found'] >= reference['found']),
                        condition(f'{regime}.moves', candidate['steps'], 1.05*reference['steps'], candidate['steps'] <= 1.05*reference['steps']),
                        condition(f'{regime}.cost80', candidate['controller_seconds'], .8*reference['controller_seconds'], candidate['controller_seconds'] <= .8*reference['controller_seconds']),
                        condition(f'{regime}.every_cost', worst, reference['controller_seconds'], worst < reference['controller_seconds'])])
        for control in ('mlp8', 'homogeneous8'):
            other = families[control]
            wins = sum(math.fsum(b[f'{control}@{s}']-b[f'min8@{s}'] for s in SEEDS)/3 > 0 for b in blocks)
            prefix = f'{regime}.{control}'
            architecture.extend([condition(prefix+'.success', candidate['found'], other['found'], candidate['found'] >= other['found']),
                                 condition(prefix+'.moves', candidate['steps'], .95*other['steps'], candidate['steps'] <= .95*other['steps']),
                                 condition(prefix+'.positive_blocks', wins, 6, wins >= 6),
                                 condition(prefix+'.cost', candidate['controller_seconds'], other['controller_seconds'], candidate['controller_seconds'] <= other['controller_seconds'])])
        regimes[regime] = {'weights': {str(h): w for h, w in weight.items()}, 'means': means, 'family_means': families,
                          'blocks': blocks, 'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in subset if r['arm'] == a and r['initial_hit'] == h)/8
                                                                    for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                          'raw_counts': {a: {'found': sum(r['found'] for r in subset if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    return {'episodes': 720, 'regimes': regimes, 'competence_checks': competence, 'compression_checks': utility,
            'architecture_checks': architecture, 'pilot_continuation': all(c['passes'] for c in competence+utility+architecture),
            'learned_architecture_advantage_established': False, 'inherited_gate_revised': False}


def payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'work-contexts.jsonl', 'work.jsonl', 'fits.jsonl', 'fit-curves.jsonl', 'epoch-orders.jsonl',
             'inference-setup.json', 'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl',
             'summary.json', 'training-costs.json', 'preparation.json', 'parity.jsonl', 'parity.json'}
    names.update(f'kernel-{r}.npz' for r in REGIMES)
    names.update(f'{split}-{suffix}' for split in ('train', 'valid') for suffix in ('data.npz', 'rows.jsonl'))
    names.update(f'{prefix}-{family}-{seed}.npz' for prefix in ('final', 'predictions') for seed in SEEDS for family in FAMILIES)
    return names


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.c = B.Comparisons()
        self.receipt = {'version': VERSION, 'status': 'started', 'scope': SCOPE, 'limits': LIMITS,
                        'training_calls': 0, 'simulator_calls': 0, 'remote_model_calls': 0,
                        'saved_checkpoint_readout_calls': 0, 'saved_network_rows': 0}

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['native_seconds']*10**9, 'native audit deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'audit output cap')

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run/'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(B.digest(path, self.check)['sha256'], pin, 'external evidence identity before decode')
        plan, worker, terminal = B.read(a.plan), B.read(a.run/'receipt.json'), B.read(a.terminal)
        for name in (RUNNER, 'scripts/audit_otto_return_value.py', 'tests/test_audit_otto_return_value.py'):
            c.equal(B.digest(ROOT/name, self.check)['sha256'], plan['sources'][name], 'source identity before lineage import')
        reader = B.load(ROOT/RUNNER, '_return_value_lineage_only')
        c.equal(reader.authenticate(a), plan, 'qualified producer lineage only')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-return-value-v1'
                and worker['completed_fits'] == 9 and worker['completed_episodes'] == 720
                and worker['external_model_calls'] == 0 and worker['pending'] == [], 'complete run before numerical decode')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker frozen plan join')
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker plan pin')
        c.equal(set(worker['files']), payload_names(), 'exact payload manifest')
        c.equal({p.name for p in a.run.iterdir()}, payload_names() | {'receipt.json'}, 'no missing late or extra payload')
        for name, expected in worker['files'].items():
            path = a.run/name
            require(path.is_file() and not path.is_symlink(), 'regular payload')
            c.equal(B.digest(path, self.check), expected, 'complete payload hashes before arrays')
        started = B.read(a.run/'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'supervision': request['supervision'], 'output': str(a.run)}, 'actual worker request')
        c.equal(B.digest(Path(request['supervision']), self.check)['sha256'], worker['supervision_sha256'], 'launch pin')
        c.equal(B.read(Path(request['supervision'])), launch, 'actual launch contents')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful complete parent before scientific decode')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            c.equal(terminal[key], launch[key], 'parent launch identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        c.equal(command[:2], [plan['python_executable'], str(ROOT/RUNNER)], 'actual interpreter/source')
        require(len(command) == 10, 'four CLI bindings')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)),
                {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'actual argument equality')
        cap = plan['limits']['native_seconds']
        require(launch['cap_seconds'] == cap and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cwd'] == str(ROOT) and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['deadline_ns'] == launch['started_ns']+cap*10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'],
                'strict native time enclosure')
        c.equal(started['started_ns'], worker['started_ns'], 'worker clock origin')
        c.equal(worker['wall_seconds'], (worker['finished_ns']-worker['started_ns'])/1e9, 'worker elapsed')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns']-terminal['started_ns'], 'parent elapsed')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns']/1e9, 'parent seconds')
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'qualified native clock')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'qualified parent source')
        require(0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'recorded resource bounds')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256, terminal_sha256=a.terminal_sha256,
                            producer_source_sha256=plan['sources'][RUNNER], authentication_reuse='Producer.authenticate lineage only.')
        return plan, worker

    def predicted(self, x, arm):
        value = predict(x, self.heads[arm], arm.split('@')[0], self.np)
        self.receipt['saved_checkpoint_readout_calls'] += 1
        self.receipt['saved_network_rows'] += len(x)
        return value

    def arrays_close(self, actual, expected, name, *, tolerance=1e-10):
        np = self.np
        require(actual.shape == expected.shape and actual.dtype == expected.dtype
                and np.isfinite(actual).all() and np.isfinite(expected).all(), name+' geometry')
        error = np.abs(actual-expected)
        maximum = float(error.max()) if error.size else 0.
        self.maximum_prediction_error = max(self.maximum_prediction_error, maximum)
        self.c.count += int(error.size)
        require(bool((error <= tolerance+tolerance*np.abs(expected)).all()), name)

    def numerical_inputs(self, plan):
        import numpy as np
        self.np = np
        self.maximum_prediction_error = 0.
        self.kernels, self.mixtures, self.datasets, self.heads = {}, {}, {}, {}
        prior = ROOT/plan['inputs']['prior_receipt']['path']
        self.prior_run = prior.parent
        for regime in REGIMES:
            with np.load(self.args.run/f'kernel-{regime}.npz', allow_pickle=False) as z:
                self.c.equal(set(z.files), {'likelihood', 'initial_hit_weights'}, 'kernel schema')
                kernel, mixture = z['likelihood'], z['initial_hit_weights']
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64 and np.isfinite(kernel).all()
                    and (kernel >= 0).all() and (kernel <= 1).all() and not kernel[:, 53, 53].any(), 'kernel geometry')
            require(mixture.shape == (4,) and mixture.dtype == np.float64 and mixture[0] == 0
                    and np.isfinite(mixture).all() and (mixture[1:] > 0).all()
                    and abs(float(mixture.sum())-1) <= 1e-12, 'initial mixture geometry')
            with np.load(self.prior_run/f'kernel-{regime}.npz', allow_pickle=False) as z:
                require(np.array_equal(kernel, z['likelihood']) and np.array_equal(mixture, z['initial_hit_weights']),
                        'all three kernels and mixtures exactly inherited')
            self.kernels[regime], self.mixtures[regime] = kernel, {h: float(mixture[h]) for h in (1, 2, 3)}
        for split in ('train', 'valid'):
            with np.load(self.args.run/f'{split}-data.npz', allow_pickle=False) as z:
                self.c.equal(set(z.files), {'features', 'target', 'beliefs', 'positions', 'sensing_length'}, 'scalar data array closure')
                data = {k: z[k] for k in z.files}
            n = len(data['target'])
            require(0 < n <= (192 if split == 'train' else 48)*HORIZON, 'complete bounded scalar data')
            for name, shape, dtype in (('features', (n, DIMENSION), np.float32), ('target', (n,), np.float32),
                                       ('beliefs', (n, 53, 53), np.float64), ('positions', (n, 2), np.int64),
                                       ('sensing_length', (n,), np.float64)):
                require(data[name].shape == shape and data[name].dtype == dtype and np.isfinite(data[name]).all(), 'scalar array shape/dtype/finiteness')
            data['rows'] = list(B.lines(self.args.run/f'{split}-rows.jsonl', self.check))
            self.c.equal([r['row_index'] for r in data['rows']], list(range(n)), 'every canonical prefix row')
            self.datasets[split] = data
        self.c.equal(len(self.datasets['train']['target']), 5589, 'all5589 teacher TRAIN prefixes')
        self.baseline = float(np.float32(self.datasets['train']['target'].mean(dtype=np.float64)))
        for seed in SEEDS:
            for kind in FAMILIES:
                with np.load(self.args.run/f'final-{kind}-{seed}.npz', allow_pickle=False) as z:
                    self.heads[f'{kind}@{seed}'] = checkpoint({k: z[k] for k in z.files}, kind, self.baseline, np)

    def reconstruct_data(self):
        np, c = self.np, self.c
        episodes = iter(B.lines(self.prior_run/'collection-episodes.jsonl', self.check))
        events = iter(B.lines(self.prior_run/'collection-transitions.jsonl', self.check))
        totals, source_rows = {'train': 0, 'valid': 0}, []
        for split, count, firsts in (('train', 96, (910001, 920001)), ('valid', 24, (930001, 940001))):
            data, index = self.datasets[split], 0
            for regime, first in zip(('lambda3', 'lambda4'), firsts, strict=True):
                for case in range(count):
                    self.check()
                    row = next(episodes)
                    seed, hit = first+case, 1+case % 3
                    episode_id = f'{split}:{regime}:{seed}:teacher'
                    identity = {'episode_id': episode_id, 'stage': split, 'regime': regime, 'seed': seed,
                                'initial_hit': hit, 'arm': 'teacher', 'block': None}
                    c.equal({k: row[k] for k in identity}, identity, 'only original teacher TRAIN/VALID identities')
                    steps = row['steps']
                    require(type(steps) is int and 1 <= steps <= HORIZON and row['found'] is True
                            and row['updates'] == steps and row['final_update_assimilated'] is True,
                            'complete naturally found teacher episode; no censor deletion')
                    public = B.packet([26, 26], hit, False, 0)
                    probability = B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels[regime], np)
                    c.equal(next(events), {'kind': 'reset', **identity, 'public': public,
                                          'posterior_after': A.posterior_record(probability),
                                          'source_evaluation_only': row['source_evaluation_only']}, 'original teacher reset')
                    target = targets(steps, np)
                    for t in range(steps):
                        state = A.posterior_record(probability)
                        metadata = {'row_index': index, 'episode_id': episode_id, 'stage': split, 'regime': regime,
                                    'seed': seed, 'initial_hit': hit, 'prefix_index': t, 'total_steps': steps,
                                    'public': public, 'posterior': state, 'target': float(target[t])}
                        c.equal(data['rows'][index], metadata, 'all exact pre-action scalar row identities')
                        require(np.array_equal(data['beliefs'][index], probability)
                                and np.array_equal(data['positions'][index], public['position'])
                                and data['sensing_length'][index] == REGIMES[regime]
                                and data['target'][index] == target[t]
                                and np.array_equal(data['features'][index], state_features(probability, public['position'], REGIMES[regime], np)),
                                'independent exact current belief/features/uncensored scalar target')
                        c.count += 5
                        event = next(events)
                        c.equal((event['kind'], event['episode_id'], event['step'], event['posterior_before']),
                                ('step', episode_id, t+1, state), 'original chronological pre-action posterior')
                        action = choice(event['costs'], public['valid_actions'], False, np)
                        c.equal(event['action'], action, 'recorded teacher eligible first tie')
                        position = B.move(public['position'], action)
                        done = position == row['source_evaluation_only']
                        require(not done or t == steps-1, 'first-found source stopping')
                        after = B.packet(position, -2 if done else event['public']['hit'], done, t+1)
                        c.equal(event['public'], after, 'original public transition geometry')
                        probability = B.posterior(probability, after, self.kernels[regime], np)
                        c.equal(event['posterior_after'], A.posterior_record(probability), 'every original public update')
                        public = after
                        index += 1
                    c.equal(public, row['final_public'], 'complete found teacher endpoint')
                    require(public['done'] is True, 'uncensored target endpoint')
                    source_rows.append({'episode_id': episode_id, 'steps': steps})
            c.equal(index, len(data['rows']), 'no sampled/extra/dropped prefix')
            totals[split] = index
        # Intentionally do not advance either stream into exposed DAgger records.
        self.data_rows = totals
        self.source_episode_rows = source_rows

    def raw_features(self, split, start, stop):
        np = self.np
        data = self.datasets[split]
        result = np.empty((stop-start, DIMENSION), np.float64)
        for j, i in enumerate(range(start, stop)):
            x, y = data['positions'][i]
            field = np.zeros((1, 105, 105), np.float64)
            field[0, 52-x:105-x, 52-y:105-y] = data['beliefs'][i]
            result[j] = features(field, data['positions'][i:i+1], float(data['sensing_length'][i]), np)[0]
        return result

    def parity(self, arm, iterator):
        np, c = self.np, self.c
        for split in ('train', 'valid'):
            data = self.datasets[split]
            for index in range(8):
                self.check()
                row = next(iterator)
                metadata = data['rows'][index]
                identity = {'fit_id': arm, 'split': split, 'row_index': index, 'episode_id': metadata['episode_id'],
                            'prefix_index': metadata['prefix_index'], 'posterior_sha256': metadata['posterior']['sha256']}
                c.equal({k: row[k] for k in identity}, identity, 'fixed parity prefixes, no replacement')
                context = {'phase': 'parity', 'fit_id': arm, 'split': split, 'row_index': index}
                for channel in ('parity_torch_forward', 'parity_numpy_forward', 'parity_torch_forward', 'parity_numpy_forward'):
                    self.work.call(channel, context)
                scalar = self.predicted(self.raw_features(split, index, index+1), arm)
                self.arrays_close(np.asarray([row['scalar_numpy']], np.float64), scalar, 'saved scalar replay')
                x, raw, weight = branches(data['beliefs'][index], data['positions'][index].tolist(),
                                           self.kernels[metadata['regime']], REGIMES[metadata['regime']], np)
                value = 64*self.predicted(x, arm)
                score = costs(value, weight, np)
                c.equal(row['raw_masses'], raw.tolist(), 'independent parity raw masses')
                c.equal(row['weights'], weight.tolist(), 'independent parity floors')
                self.arrays_close(np.asarray(row['values_numpy'], np.float64), value, 'saved parity values')
                self.arrays_close(np.asarray(row['costs_numpy'], np.float64), score, 'saved parity costs')
                c.equal(row['allowed_actions'], metadata['public']['valid_actions'], 'parity public eligible mask')
                for name in ('values', 'costs'):
                    self.arrays_close(np.asarray(row[name+'_numpy'], np.float64), np.asarray(row[name+'_torch'], np.float64),
                                      'saved Torch-double comparison/'+name)
                self.arrays_close(np.asarray([row['scalar_numpy']], np.float64), np.asarray([row['scalar_torch']], np.float64),
                                  'saved Torch-double scalar comparison')
                expected = choice(score.tolist(), row['allowed_actions'], True, np)
                c.equal(row['action_numpy'], choice(row['costs_numpy'], row['allowed_actions'], True, np), 'saved NumPy parity first tie')
                c.equal(row['action_torch'], choice(row['costs_torch'], row['allowed_actions'], True, np), 'saved Torch parity first tie')
                c.equal((row['action_numpy'], row['action_torch'], row['passed']), (expected, expected, True), 'exact original parity acceptance')

    def fit_records(self):
        np, c = self.np, self.c
        fits = iter(B.lines(self.args.run/'fits.jsonl', self.check))
        curves = iter(B.lines(self.args.run/'fit-curves.jsonl', self.check))
        orders = iter(B.lines(self.args.run/'epoch-orders.jsonl', self.check))
        parity = iter(B.lines(self.args.run/'parity.jsonl', self.check))
        train = self.datasets['train']
        metadata = [{k: v for k, v in row.items() if k != 'row_index'} for row in train['rows']]
        row_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        array_hashes = {key: hashlib.sha256(train[key].tobytes()).hexdigest() for key in ('features', 'target')}
        self.fit_seconds, self.final_diagnostics = 0., {}
        ntrain, nvalid = self.data_rows['train'], self.data_rows['valid']
        for seed in SEEDS:
            for family in FAMILIES:
                arm = f'{family}@{seed}'
                row = next(fits)
                expected = {'fit_id': arm, 'family': family, 'seed': seed, 'epochs': 80, 'training_rows': ntrain,
                            'validation_rows': nvalid, 'training_episodes': 192, 'training_array_sha256': array_hashes,
                            'row_order_sha256': row_hash, 'c0_float32': self.baseline,
                            'optimizer_steps': [3520]*len(tensor_names(family)),
                            'parameter_count': {'min8': 88224, 'mlp8': 88241, 'homogeneous8': 88232}[family]}
                c.equal({k: row[k] for k in expected}, expected, 'all nine fixed scalar fits and uniform exposure')
                c.equal(row['optimizer_steps_before'], [], 'fresh optimizer in every fit')
                c.equal(set(row['initial_array_sha256']), {'c0', *tensor_names(family)}, 'complete initial state hashes')
                require(row['initial_biases_zero'] is True, 'zero initial bias witness')
                initial = self.initial_hashes.setdefault(seed, {})
                for key in ('c0', 'first_weight', 'final_weight'):
                    if key in row['initial_array_sha256']:
                        pin = row['initial_array_sha256'][key]
                        require(isinstance(pin, str) and len(pin) == 64 and all(x in '0123456789abcdef' for x in pin), 'initial tensor hash')
                        if key in initial:
                            c.equal(pin, initial[key], 'paired same-seed initial tensor witness')
                        initial[key] = pin
                rng = np.random.default_rng(seed+20000)
                paid, curve = [], []
                for epoch in range(1, 81):
                    self.check()
                    order = rng.permutation(ntrain)
                    c.equal(next(orders), {'fit_id': arm, 'epoch': epoch, 'rows': ntrain,
                                          'sha256': hashlib.sha256(order.tobytes()).hexdigest()}, 'all720 independently reconstructed epoch permutations')
                    for batch in range(math.ceil(ntrain/128)):
                        context = {'phase': 'fit', 'fit_id': arm, 'epoch': epoch, 'batch': batch}
                        paid.append(self.work.call('optimizer_update', context)['seconds'])
                    if epoch % 10 == 0:
                        for _ in range(math.ceil(nvalid/128)):
                            paid.append(self.work.call('validation_forward', context)['seconds'])
                        record = next(curves)
                        c.equal(set(record), {'fit_id', 'epoch', 'training_mse_normalized', 'validation_mse_normalized'}, 'intermediate curve schema')
                        c.equal((record['fit_id'], record['epoch']), (arm, epoch), 'eight descriptive curves perfit')
                        require(all(math.isfinite(record[k]) and record[k] >= 0 for k in ('training_mse_normalized', 'validation_mse_normalized')),
                                'finite inherited intermediate training/validation witnesses')
                        curve.append(record)
                c.equal(row['curve'], curve, 'all72 descriptive curves retained')
                context = {'phase': 'fit_export', 'fit_id': arm}
                with np.load(self.args.run/f'predictions-{family}-{seed}.npz', allow_pickle=False) as z:
                    c.equal(set(z.files), {'train', 'valid'}, 'complete final scalar predictions')
                    diagnostics = {}
                    for split in ('train', 'valid'):
                        saved = z[split]
                        data = self.datasets[split]
                        require(saved.dtype == np.float64 and saved.shape == data['target'].shape and np.isfinite(saved).all(), 'finite complete final scalar arrays')
                        rebuilt = np.empty_like(saved)
                        planes = np.zeros(8, np.int64) if family == 'min8' else None
                        for offset in range(0, len(saved), 128):
                            self.check()
                            paid.append(self.work.call('saved_value_forward', context)['seconds'])
                            end = min(offset+128, len(saved))
                            x = self.raw_features(split, offset, end)
                            rebuilt[offset:end] = self.predicted(x, arm)
                            if planes is not None:
                                paid.append(self.work.call('plane_diagnostic', context)['seconds'])
                                selected = (x @ self.heads[arm]['first_weight'].T).argmin(axis=1)
                                planes += np.bincount(selected, minlength=8)
                                self.receipt['saved_plane_diagnostic_calls'] = self.receipt.get('saved_plane_diagnostic_calls', 0)+1
                        self.arrays_close(saved, rebuilt, 'all final TRAIN/VALID scalar predictions')
                        diagnostics[split] = {'mse_normalized': float(np.mean((rebuilt-data['target'])**2)),
                                              'negative_count': int((rebuilt < 0).sum()), 'rows': len(rebuilt),
                                              'negative_fraction': float((rebuilt < 0).mean()),
                                              'plane_counts': planes.tolist() if planes is not None else None}
                    c.tree(row['diagnostics'], diagnostics, 'independent final prediction diagnostics')
                    self.final_diagnostics[arm] = diagnostics
                self.parity(arm, parity)
                filename = f'final-{family}-{seed}.npz'
                c.equal((row['checkpoint'], row['checkpoint_sha256'], row['parity_records'], row['parity_passed']),
                        (filename, B.digest(self.args.run/filename)['sha256'], 16, True), 'fixed complete final checkpoint and parity')
                require(math.isfinite(row['fit_seconds']) and row['fit_seconds'] >= math.fsum(paid)-1e-8, 'recorded fit encloses paid work')
                self.fit_seconds += row['fit_seconds']
        for iterator, name in ((fits, 'fits'), (curves, 'curves'), (orders, 'epoch orders'), (parity, 'parity')):
            B.exhausted(iterator, name)
        saved = B.read(self.args.run/'parity.json')
        c.equal(saved, {'status': 'completed', 'records': 144, 'fits': 9, 'atol': 1e-10, 'rtol': 1e-10,
                        'all_passed': True, 'exact_action_required': True,
                        'states_per_fit': [{'split': split, 'row_index': i} for split in ('train', 'valid') for i in range(8)]},
                'complete parity gate before evaluation')

    def qualification(self):
        np, c = self.np, self.c
        native = B.read(self.args.run/'native-setup.json')
        c.equal(native['native_resets'], 3, 'all three fresh kernel templates')
        c.equal(len(native['checks']), 3, 'all template witnesses')
        for index, regime in enumerate(REGIMES):
            self.work.call('native_reset', {'phase': 'native_setup', 'regime': regime})
            row = native['checks'][index]
            c.equal((row['regime'], row['template_seed'], row['old_kernel_exact']),
                    (regime, 11500001+index, True), 'exact inherited sensor and prior template witness')
            require(row['initial_public']['position'] == [26, 26] and row['initial_public']['step'] == 0
                    and row['initial_public']['done'] is False and row['initial_public']['hit'] in (1, 2, 3), 'template reset packet')
        events = iter(B.lines(self.args.run/'qualification.jsonl', self.check))
        checks, total = [], 0
        for case in range(8):
            context = {'phase': 'qualification', 'case': case}
            self.work.call('native_reset', context)
            row = next(events)
            public = B.packet([26, 26], 1+case % 3, False, 0)
            c.equal((row['kind'], row['case'], row['public']), ('reset', case, public), 'prescribed qualification reset')
            source = row['source_evaluation_only']
            probability = B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels['lambda5'], np)
            for step in range(1, 33):
                self.work.call('native_step', {**context, 'step': step})
                event = next(events)
                action = (0, 2, 1, 3)[(step-1) % 4]
                target = B.move(public['position'], action)
                found = target == source
                c.equal((event['kind'], event['case'], event['step'], event['action']), ('step', case, step, action), 'all prescribed qualification actions')
                after = B.packet(target, -2 if found else event['public']['hit'], found, step)
                c.equal(event['public'], after, 'qualification movement and found event')
                probability = B.posterior(probability, after, self.kernels['lambda5'], np)
                c.equal(event['posterior'], A.posterior_record(probability), 'qualification public filter')
                public = after
                total += 1
                if found:
                    break
            checks.append({'case': case, 'steps': step, 'found': found, 'passed': True})
        B.exhausted(events, 'qualification events')
        saved = B.read(self.args.run/'qualification.json')
        c.equal(saved['checks'], checks, 'all8 complete fresh qualification witnesses')
        c.equal((saved['status'], saved['resets'], saved['native_steps']), ('completed', 8, total), 'qualification work totals')
        return total

    def episode(self, row, identity, events):
        np, c = self.np, self.c
        regime, seed, hit, arm, block = identity
        episode_id = f'eval:{regime}:{seed}:{arm}'
        canonical = {'episode_id': episode_id, 'stage': 'eval', 'regime': regime, 'seed': seed,
                     'initial_hit': hit, 'arm': arm, 'block': block}
        c.equal({k: row[k] for k in canonical}, canonical, 'all720 evaluation identities')
        steps = row['steps']
        require(type(steps) is int and 1 <= steps <= HORIZON and type(row['found']) is bool
                and (row['found'] or steps == HORIZON), 'complete found/censored duration')
        context = {'phase': 'eval', 'episode': episode_id, 'step': 0}
        self.work.call('native_reset', context)
        public = B.packet([26, 26], hit, False, 0)
        source = row['source_evaluation_only']
        require(len(source) == 2 and all(type(v) is int and 0 <= v < 53 for v in source)
                and source != [26, 26], 'evaluator source geometry')
        probability = B.posterior(np.ones((53, 53), np.float64)/2808, public, self.kernels[regime], np)
        state = A.posterior_record(probability)
        c.equal(next(events), {'kind': 'reset', **canonical, 'public': public, 'posterior_after': state,
                               'source_evaluation_only': source}, 'evaluation initial public state')
        pair = (regime, seed)
        if pair in self.sources:
            c.equal((source, public), self.sources[pair], 'paired sampled source and reset')
        else:
            self.sources[pair] = (source, public)
        draws = iter(row['draws_evaluation_only'])
        first = next(draws)
        c.equal((first['channel'], first['index'], first['selected_index']), ('source', 0, 53*source[0]+source[1]), 'source draw identity')
        learned = arm != 'analytic_inbounds'
        totals = {key: [] for key in ('choose_seconds', 'update_seconds', 'environment_seconds', 'choose_instrumented_seconds', 'choose_excluded_io_seconds')}
        positions, zero = [], 0
        for step in range(1, steps+1):
            self.check()
            ctx = {**context, 'step': step}
            operation = self.work.call('value_forward' if learned else 'analytic_choose', ctx)
            native = self.work.call('native_step', ctx)
            event = next(events)
            c.equal((event['kind'], event['episode_id'], event['step']), ('step', episode_id, step), 'every ordered evaluation decision')
            allowed = public['valid_actions']
            c.equal(event['allowed_actions'], allowed, 'public-only action mask')
            action = choice(event['costs'], allowed, learned, np)
            c.equal(event['action'], action, 'exact first eligible recorded-cost action')
            if learned:
                x, raw, weight = branches(probability, public['position'], self.kernels[regime], REGIMES[regime], np)
                c.equal(event['raw_masses'], raw.tolist(), 'all sixteen independent raw branch masses')
                c.equal(event['weights'], weight.tolist(), 'all sixteen independent exact floors')
                value = 64*self.predicted(x, arm)
                rebuilt = costs(value, weight, np)
                self.arrays_close(np.asarray(event['values'], np.float64), value, 'independent physical branch values')
                self.arrays_close(np.asarray(event['costs'], np.float64), rebuilt, 'independent explicit four costs')
                c.equal(action, choice(rebuilt.tolist(), allowed, True, np), 'independent exact selected action; no tie exemption')
            else:
                c.equal((event['raw_masses'], event['weights'], event['values']), (None, None, None), 'analytic has no neural telemetry')
            positions.append(tuple(public['position']))
            zero += state['mass'] == 0
            target = B.move(public['position'], action)
            found = target == source
            require(target != public['position'] and (not found or step == steps), 'inbounds motion and immediate found stopping')
            observed = event['public']['hit']
            require(type(observed) is int and (observed == -2 if found else 0 <= observed < 4), 'hit category or terminal sentinel')
            after = B.packet(target, observed, found, step)
            c.equal(event['public'], after, 'independent public transition')
            c.equal(event['native_p_end'], float(found), 'sampled native termination')
            c.equal(event['posterior_before'], state, 'before-action public posterior')
            probability = B.posterior(probability, after, self.kernels[regime], np)
            state = A.posterior_record(probability)
            c.equal(event['posterior_after'], state, 'all updates including final found/censor')
            if not found:
                draw = next(draws)
                c.equal((draw['channel'], draw['index'], draw['selected_index']), ('hit', step-1, observed), 'chronological public hit draw')
            for key, values in totals.items():
                value = event[key]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative operation timing')
                values.append(value)
            c.close(event['environment_seconds'], native['seconds'], 'environment operation cost')
            c.close(event['choose_seconds'], event['choose_instrumented_seconds']-event['choose_excluded_io_seconds'], 'only recorded I/O excluded')
            require(event['choose_seconds']+1e-9 >= operation['seconds'], 'complete branch/feature/readout/mask duration encloses scalar operation')
            public = after
        B.exhausted(draws, 'episode draw sequence')
        for draw in row['draws_evaluation_only']:
            require(0 <= draw['uniform'] < 1 and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10, 'finite recorded random draw')
            key = (*pair, draw['channel'], draw['index'])
            if key in self.uniforms:
                c.equal(draw['uniform'], self.uniforms[key], 'paired channel-index uniform witness')
            else:
                self.uniforms[key] = draw['uniform']
        c.equal(row['final_public'], public, 'complete final packet')
        c.equal((row['found'], row['updates'], row['blocked_steps'], row['final_update_assimilated']),
                (public['done'], steps, 0, True), 'all complete outcomes/updates retained')
        tail = positions[-256:]
        diagnostic = (zero, state['mass'], sum(tail[i] == tail[i-2] for i in range(2, len(tail))),
                      max(0, len(tail)-2), len(set(positions)))
        c.equal(tuple(row[k] for k in ('zero_mass_decisions', 'final_posterior_mass', 'last256_lag2_matches', 'last256_lag2_pairs', 'distinct_preaction_positions')),
                diagnostic, 'descriptive mass and spatial repetition only')
        for key, values in totals.items():
            c.close(row[key], math.fsum(values), 'summed full episode operation costs')
        c.close(row['setup_allocation_seconds'], 0., 'physical episode before setup allocation')
        c.close(row['controller_seconds'], math.fsum(row[k] for k in TIMES), 'complete physical controller cost')
        require(math.isfinite(row['init_seconds']) and row['init_seconds'] >= 0, 'finite controller initialization')
        c.equal(row['state_bytes'], 22472, 'full public belief state storage')
        storage = row['storage']
        c.equal(set(storage), {'public_actor', 'head', 'branch_workspace'}, 'all retained storage components')
        c.equal({k: v for k, v in storage['public_actor'].items() if k != 'scope'},
                {'immutable_array_bytes': 366368+91592, 'mutable_array_bytes': 22472,
                 'immutable_arrays': {'observation_kernel': 366368, 'manhattan_distance_table': 91592}}, 'including inherited unused distance table')
        c.equal({k: v for k, v in storage['head'].items() if k != 'scope'} if learned else storage['head'],
                head_storage(arm.split('@')[0]) if learned else None, 'actual float64 runtime head storage')
        c.equal(storage['branch_workspace'], '16*105*105 float64 centered u and z plus finite branch/features workspace, transient', 'transient workspace disclosure')
        return row

    def compute(self, plan, worker):
        self.numerical_inputs(plan)
        c = self.c
        runtime = B.read(self.args.run/'runtime.json')
        c.equal(runtime['executable'], plan['python_executable'], 'recorded runtime interpreter')
        c.equal(runtime['versions'], plan['runtime_versions'], 'recorded numerical versions')
        c.equal((runtime['torch_threads'], runtime['torch_interop_threads'], runtime['module_allocation_episodes']),
                (1, 1, 648), 'CPU1 and all learned module setup recipients')
        c.equal(runtime['environment'], {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'fixed thread controls')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            require(math.isfinite(runtime[key]) and runtime[key] >= 0, 'finite distinct setup durations')
        self.reconstruct_data()
        c.equal(B.read(self.args.run/'preparation.json'),
                {'status': 'completed', 'source_receipt_sha256': plan['inputs']['prior_receipt']['sha256'],
                 'c0_float32': self.baseline, 'episodes': {'train': 192, 'valid': 48}, 'rows': self.data_rows,
                 'all_found': True, 'all_prefixes': True, 'dagger_or_evaluation_decoded': False, 'weighting': 'uniform row',
                 'data': {split: B.digest(self.args.run/f'{split}-data.npz') for split in ('train', 'valid')}},
                'all uncensored teacher prefixes and TRAIN-only f32 baseline')
        contexts = list(B.lines(self.args.run/'work-contexts.jsonl', self.check))
        c.equal([r['id'] for r in contexts], list(range(len(contexts))), 'contiguous work contexts')
        require(all(set(r) == {'id', 'context'} and 'step' not in r['context'] for r in contexts)
                and len({json.dumps(r['context'], sort_keys=True) for r in contexts}) == len(contexts), 'unique work contexts')
        self.work = Work(iter(B.lines(self.args.run/'work.jsonl', self.check)), [r['context'] for r in contexts], c)
        qualification_steps = self.qualification()
        self.initial_hashes = {}
        self.fit_records()
        setup = B.read(self.args.run/'inference-setup.json')
        c.equal(set(setup), set(ARMS[:-1]), 'nine distinct runtime head loads')
        for arm, record in setup.items():
            family, seed = arm.split('@')
            filename = f'final-{family}-{seed}.npz'
            c.equal((record['checkpoint'], record['sha256'], record['allocated_episodes']),
                    (filename, B.digest(self.args.run/filename)['sha256'], 72), 'each final head allocated over72 episodes')
            require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'finite head restoration cost')
            c.equal({k: v for k, v in record['storage'].items() if k != 'scope'}, head_storage(family), 'owned f64 head arrays and baseline')
        episodes = iter(B.lines(self.args.run/'eval-episodes.jsonl', self.check))
        allocated = iter(B.lines(self.args.run/'evaluation.jsonl', self.check))
        events = iter(B.lines(self.args.run/'eval-transitions.jsonl', self.check))
        self.sources, self.uniforms = {}, {}
        rows = []
        for identity in evaluation_order():
            row = self.episode(next(episodes), identity, events)
            arm = row['arm']
            allocation = setup[arm]['seconds']/72 + runtime['model_module_setup_seconds']/648 if arm in setup else 0.
            completed = {**row, 'setup_allocation_seconds': allocation, 'controller_seconds': row['controller_seconds']+allocation}
            c.equal(next(allocated), completed, 'full cost plus actual head/module amortization')
            rows.append({k: v for k, v in completed.items() if k != 'draws_evaluation_only'})
        for iterator, name in ((episodes, 'physical evaluation'), (allocated, 'allocated evaluation'), (events, 'complete evaluation events'),
                               (self.work.iterator, 'complete operation journal')):
            B.exhausted(iterator, name)
        c.equal(self.work.used_contexts, set(range(len(contexts))), 'no unused contexts')
        c.tree(worker['calls'], self.work.counts, 'all attempted/returned operation counts and times')
        train_batches, valid_batches = math.ceil(self.data_rows['train']/128), math.ceil(self.data_rows['valid']/128)
        expected_calls = {'native_reset': 731, 'native_step': qualification_steps+sum(r['steps'] for r in rows),
                          'optimizer_update': 9*80*train_batches, 'validation_forward': 9*8*valid_batches,
                          'saved_value_forward': 9*(train_batches+valid_batches), 'plane_diagnostic': 3*(train_batches+valid_batches),
                          'parity_torch_forward': 288, 'parity_numpy_forward': 288,
                          'value_forward': sum(r['steps'] for r in rows if r['arm'] != 'analytic_inbounds'),
                          'analytic_choose': sum(r['steps'] for r in rows if r['arm'] == 'analytic_inbounds')}
        c.equal({k: v['returned'] for k, v in self.work.counts.items()}, expected_calls, 'all native/model/optimizer work covered')
        c.equal((worker['prepared_episodes'], worker['training_updates'], worker['completed_fits'], worker['completed_episodes']),
                (240, 31680, 9, 720), 'complete fixed exposure and execution')
        for channel, limit in (('native_step', 'native_steps'), ('native_reset', 'native_resets'), ('optimizer_update', 'optimizer_updates')):
            require(expected_calls[channel] <= plan['limits'][limit], 'prospective operation cap')
        derived, published = aggregate(rows, self.mixtures), B.read(self.args.run/'summary.json')
        for key, value in derived.items():
            c.tree(published[key], value, 'independent summary/'+key)
        c.equal(published['version'], 'otto-return-value-v1', 'summary experiment identity')
        c.tree(published['calls'], self.work.counts, 'summary recorded work')
        c.equal(published['inference_setup'], setup, 'complete summary runtime setup')
        for key in ('shared_setup_seconds', 'model_module_setup_seconds'):
            c.equal(published[key], runtime[key], 'disjoint setup publication')
        c.equal((published['model_module_allocation_episodes'], published['paired_source_cases'], published['paired_uniforms'],
                 published['all_public_beliefs_exact']), (648, 72, len(self.uniforms), True), 'complete public pair/filter witnesses')
        c.equal(worker['pilot_continuation'], derived['pilot_continuation'], 'no reversal of54-condition outcome')
        training = B.read(self.args.run/'training-costs.json')
        c.equal(set(training), {'saved_teacher_preparation_seconds', 'fitting_and_parity_seconds'}, 'separate saved preparation and fitting costs')
        require(all(math.isfinite(v) and v >= 0 for v in training.values())
                and training['fitting_and_parity_seconds'] >= self.fit_seconds-1e-8, 'finite enclosing phase costs')
        evaluation_cost = math.fsum(r['controller_seconds']+r['environment_seconds'] for r in rows)
        disjoint = math.fsum(training.values())+evaluation_cost+runtime['shared_setup_seconds']
        require(disjoint <= worker['wall_seconds']+1e-8, 'disjoint paid phases within worker interval')
        require(math.fsum(r['choose_excluded_io_seconds'] for r in rows) <= worker['artifact_io_seconds']+1e-8,
                'excluded I/O bounded by total recorded artifact work')
        return {'version': VERSION, 'agreement': True, **derived, 'comparisons': c.count,
                'maximum_scalar_difference': c.maximum_difference, 'maximum_prediction_difference': self.maximum_prediction_error,
                'dataset_rows': self.data_rows, 'teacher_episodes': 240, 'c0_float32': self.baseline,
                'final_prediction_diagnostics': self.final_diagnostics, 'fits': 9, 'epoch_orders': 720,
                'descriptive_validation_records': 72, 'parity_records': 144, 'qualified_resets': 11,
                'qualification_updates': qualification_steps, 'evaluation_updates': sum(r['steps'] for r in rows),
                'actual_work': expected_calls, 'saved_checkpoint_readout_calls': self.receipt['saved_checkpoint_readout_calls'],
                'saved_network_rows': self.receipt['saved_network_rows'],
                'condition_counts': {'competence': 18, 'utility_compute': 12, 'architecture': 24},
                'costs': {'training_phases': training, 'inference_setup': setup,
                          'shared_setup_seconds': runtime['shared_setup_seconds'], 'model_module_setup_seconds': runtime['model_module_setup_seconds'],
                          'disjoint_accounted_seconds': disjoint, 'worker_seconds': worker['wall_seconds']},
                'paired_cases': [{k: r[k] for k in ('regime', 'seed', 'arm', 'block', 'initial_hit', 'steps', 'found', 'controller_seconds',
                                                  'zero_mass_decisions', 'last256_lag2_matches', 'last256_lag2_pairs')} for r in rows], 'scope': SCOPE}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents),
                'exclusive absolute audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(B.digest(ROOT/B.CLOCK)['sha256'] == B.CLOCK_PIN, 'qualified suspend-inclusive clock')
            self.clock = B.load(ROOT/B.CLOCK, '_return_value_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit emergency cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['native_seconds'])
            self.receipt.update(source=B.digest(Path(__file__), self.check), independent_helper={'path': str(HELPER), 'sha256': HELPER_PIN})
            B.write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()}, 'limits': LIMITS,
                                            'clock_backend': self.clock.backend, 'started_ns': self.start})
            plan, worker = self.authenticate()
            result = self.compute(plan, worker)
            self.authenticate()
            B.write(self.out/'summary.json', result)
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', agreement=True, clock_backend=self.clock.backend, started_ns=self.start,
                                finished_ns=finished, wall_seconds=(finished-self.start)/1e9, comparisons=result['comparisons'],
                                files={p.name: B.digest(p, self.check) for p in self.out.iterdir()})
            B.write(self.out/'receipt.json', self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                B.write(self.out/'failed.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Preserve primary failure.
                error.add_note(f'Failure preservation: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


def head_storage(kind):
    return {'parameter_array_bytes': {'min8': 88224, 'mlp8': 88241, 'homogeneous8': 88232}[kind]*8,
            'baseline_array_bytes': 8, 'mutable_array_bytes': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
