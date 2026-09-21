"""Independent saved-output audit for the prospective full-belief symmetry pilot.

No training, simulator or remote calls. Local NumPy checkpoint readouts are
explicitly counted. Frozen independent public-filter primitives are reused;
producer feature/readout/metric functions are not used for arithmetic.
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
BASE_PATH = ROOT / 'scripts/audit_otto_released_reference.py'
BASE_PIN = '6f63cbc79107fc3046d28645fa4336fe04bfb91ee53368c78ba3971ced1942e1'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_PIN:
    raise ValueError('frozen independent public-filter source identity')
_spec = importlib.util.spec_from_file_location('_symmetry_independent_filter', BASE_PATH)
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B
_spec.loader.exec_module(B)
require = B.require

VERSION = 'otto-symmetry-head-saved-audit-v1'
LIMITS = {'native_seconds': 1800, 'rss_bytes': 8 * 1024**3, 'output_bytes': 512 * 1024**2}
SIZE, DIMENSION, HORIZON = 53, 2836, 2188
FIT_SEEDS = (9101, 9102, 9103)
FAMILIES = ('d4_shared', 'dense_augmented', 'dense_ensemble')
WEIGHT_NAMES = tuple(f'{kind}{i}' for i in range(3) for kind in ('weight', 'bias'))


def selected_prefixes(steps):
    require(type(steps) is int and 1 <= steps <= HORIZON, 'complete bounded collection episode')
    count = min(64, steps)
    return [0] if count == 1 else [round(i * (steps - 1) / (count - 1)) for i in range(count)]


def feature_vector(probability, public, kernel, sensing_length, np):
    """Public belief and twenty unnormalized local forecasts, no lookahead."""
    require(set(public) == {'position', 'hit', 'done', 'step', 'valid_actions'}
            and public['done'] is False, 'exact live public packet')
    position = public['position']
    require(len(position) == 2 and all(type(v) is int and 0 <= v < SIZE for v in position), 'public position')
    valid = [a for a in range(4) if B.move(position, a) != position]
    require(public['valid_actions'] == valid and sensing_length in (3., 4., 5.), 'known public model and legal actions')
    require(probability.shape == (SIZE, SIZE) and probability.dtype == np.float64
            and np.isfinite(probability).all() and (probability >= 0).all(), 'finite nonnegative exact public belief')
    forecast = []
    for action in range(4):
        x, y = B.move(position, action)
        forecast.extend([float(probability[x, y]),
                         *(float((probability * kernel[h, SIZE-x:2*SIZE-x, SIZE-y:2*SIZE-y]).sum()) for h in range(4))])
    value = np.concatenate((SIZE * np.sqrt(probability).ravel(),
                            [position[0] / 26 - 1, position[1] / 26 - 1],
                            [float(a in valid) for a in range(4)], [sensing_length / 5], forecast)).astype(np.float32)
    require(value.shape == (DIMENSION,) and np.isfinite(value).all(), 'fixed finite public feature vector')
    return value


def action_permutation(group):
    require(type(group) is int and 0 <= group < 8, 'fixed D4 element')
    vectors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    result = []
    for x, y in vectors:
        if group >= 4:
            y = -y
        for _ in range(group % 4):
            x, y = -y, x
        result.append(vectors.index((x, y)))
    return result


def transformed_features(features, group, np):
    """Independent coordinate transform, including action-major forecasts."""
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == DIMENSION,
            'batched float32 features')
    perm = action_permutation(group)
    result = features.copy()
    field = features[:, :SIZE*SIZE].reshape(-1, SIZE, SIZE)
    if group >= 4:
        field = field[:, :, ::-1]
    result[:, :SIZE*SIZE] = np.rot90(field, group % 4, axes=(1, 2)).reshape(-1, SIZE*SIZE)
    x, y = features[:, SIZE*SIZE].copy(), features[:, SIZE*SIZE+1].copy()
    if group >= 4:
        y = -y
    for _ in range(group % 4):
        x, y = -y, x
    result[:, SIZE*SIZE], result[:, SIZE*SIZE+1] = x, y
    result[:, np.asarray(perm) + SIZE*SIZE+2] = features[:, SIZE*SIZE+2:SIZE*SIZE+6]
    forecasts = result[:, -20:].reshape(-1, 4, 5)
    forecasts[:, perm, :] = features[:, -20:].reshape(-1, 4, 5)
    return result


def validate_weights(weights, kind, np):
    require(kind in ('d4_shared', 'dense_augmented') and set(weights) == set(WEIGHT_NAMES), 'exact six trained tensors')
    shapes = ((32, DIMENSION), (32,), (16, 32), (16,), (1 if kind == 'd4_shared' else 4, 16), (1 if kind == 'd4_shared' else 4,))
    for key, shape in zip(WEIGHT_NAMES, shapes, strict=True):
        value = weights[key]
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), 'finite exact checkpoint tensors')


def network(features, weights, np):
    hidden = features
    for layer in range(3):
        hidden = hidden @ weights[f'weight{layer}'].T + weights[f'bias{layer}']
        if layer < 2:
            hidden = np.tanh(hidden)
    require(hidden.dtype == np.float32 and np.isfinite(hidden).all(), 'finite independently replayed costs')
    return hidden


def readout(features, weights, mode, np):
    """Match scalar/batched BLAS geometry without producer readout helpers."""
    require(mode in FAMILIES and features.ndim in (1, 2), 'fixed learned inference mode')
    if mode == 'dense_augmented':
        return network(features, weights, np)
    single = features.ndim == 1
    batch = features[None] if single else features
    views = np.stack([transformed_features(batch, g, np) for g in range(8)], axis=1)
    costs = network(views[0] if single else views, weights, np)
    if mode == 'dense_ensemble':
        ordered = np.stack([costs[..., g, action_permutation(g)] for g in range(8)], axis=-2)
        return ordered.mean(axis=-2, dtype=np.float32)
    pairs = [[g for g in range(8) if action_permutation(g)[a] == 0] for a in range(4)]
    return costs[..., 0][..., pairs].mean(axis=-1, dtype=np.float32)


def teacher_target(costs, valid, np):
    require(len(costs) == len(valid) == 4 and any(valid), 'teacher support')
    require(all((costs[a] is not None and math.isfinite(costs[a])) if valid[a] else costs[a] is None for a in range(4)),
            'teacher inbounds cost convention')
    selected = np.asarray([costs[a] for a in range(4) if valid[a]], np.float64)
    scaled = -(selected - selected.min()) / max(float(selected.max() - selected.min()), 1e-8) / .25
    probability = np.exp(scaled - scaled.max())
    probability /= probability.sum()
    result = np.zeros(4, np.float32)
    result[np.asarray(valid)] = probability
    return result


def balanced_weights(episode_ids, np):
    require(len(episode_ids) > 0, 'nonempty TRAIN membership')
    counts = {}
    for key in episode_ids:
        counts[key] = counts.get(key, 0) + 1
    return np.asarray([len(episode_ids) / (len(counts) * counts[k]) for k in episode_ids], np.float32)


ARM_FAMILIES = ('shared', 'dense', 'dense_ensemble')
ARMS = tuple(f'{f}@{s}' for s in FIT_SEEDS for f in ARM_FAMILIES) + ('analytic_inbounds',)
REGIMES = {'lambda3': 3., 'lambda4': 4., 'lambda5': 5.}
FIRST = {'train': {'lambda3': 910001, 'lambda4': 920001},
         'valid': {'lambda3': 930001, 'lambda4': 940001},
         'dagger': {'lambda3': 950001, 'lambda4': 960001},
         'eval': {'lambda3': 970001, 'lambda4': 980001, 'lambda5': 990001}}
TIMES = ('init_seconds', 'choose_seconds', 'update_seconds', 'setup_allocation_seconds')
METRICS = ('steps', 'found', *TIMES, 'controller_seconds', 'environment_seconds', 'state_bytes')


def episode_order(stage):
    require(stage in FIRST, 'fixed data stage')
    if stage == 'eval':
        for ri, (regime, first) in enumerate(FIRST[stage].items()):
            for case in range(24):
                shift = (ri*24 + case) % 10
                for arm in ARMS[shift:] + ARMS[:shift]:
                    yield regime, first+case, 1+case % 3, arm, case//3
    else:
        arms = [f'{f}@{s}' for s in FIT_SEEDS for f in ('shared', 'dense')] if stage == 'dagger' else ['teacher']
        for regime, first in FIRST[stage].items():
            for arm in arms:
                for case in range({'train': 96, 'valid': 24, 'dagger': 12}[stage]):
                    yield regime, first+case, 1+case % 3, arm, None


def condition(name, value, threshold, passes):
    return {'name': name, 'value': value, 'threshold': threshold, 'passes': bool(passes)}


def aggregate(rows, mixtures):
    require([(r['regime'], r['seed'], r['initial_hit'], r['arm'], r['block']) for r in rows]
            == list(episode_order('eval')), 'all720 exact evaluation identities and rotation')
    regimes, competence, compression, architecture = {}, [], [], []
    for regime in REGIMES:
        weight = {int(h): float(w) for h, w in mixtures[regime].items()}
        require(set(weight) == {1, 2, 3} and all(math.isfinite(w) and 0 < w < 1 for w in weight.values())
                and abs(math.fsum(weight.values())-1) <= 1e-12, 'complete positive evaluation mixture')
        subset = [r for r in rows if r['regime'] == regime]

        def mean(selected, metric, weight=weight):
            parts = [[float(r[metric]) for r in selected if r['initial_hit'] == h] for h in (1, 2, 3)]
            require(all(parts), 'all three conditional strata present')
            return math.fsum(weight[h] * math.fsum(v)/len(v) for h, v in zip((1, 2, 3), parts, strict=True))

        means = {a: {m: mean([r for r in subset if r['arm'] == a], m) for m in METRICS} for a in ARMS}
        blocks = [{a: mean([r for r in subset if r['arm'] == a and r['block'] == b], 'steps') for a in ARMS} for b in range(8)]
        family = {f: {m: math.fsum(means[f'{f}@{seed}'][m] for seed in FIT_SEEDS)/3 for m in METRICS} for f in ARM_FAMILIES}
        reference, shared = means['analytic_inbounds'], family['shared']
        for seed in FIT_SEEDS:
            fit = means[f'shared@{seed}']
            competence.extend([condition(f'{regime}.{seed}.success', fit['found'], .95, fit['found'] >= .95),
                               condition(f'{regime}.{seed}.moves', fit['steps'], 1.05*reference['steps'], fit['steps'] <= 1.05*reference['steps'])])
        worst = max(means[f'shared@{seed}']['controller_seconds'] for seed in FIT_SEEDS)
        compression.extend([condition(f'{regime}.success', shared['found'], reference['found'], shared['found'] >= reference['found']),
                            condition(f'{regime}.moves', shared['steps'], 1.05*reference['steps'], shared['steps'] <= 1.05*reference['steps']),
                            condition(f'{regime}.cost80', shared['controller_seconds'], .8*reference['controller_seconds'], shared['controller_seconds'] <= .8*reference['controller_seconds']),
                            condition(f'{regime}.every_cost', worst, reference['controller_seconds'], worst < reference['controller_seconds'])])
        for control in ('dense', 'dense_ensemble'):
            other = family[control]
            wins = sum(math.fsum(block[f'{control}@{seed}']-block[f'shared@{seed}'] for seed in FIT_SEEDS)/3 > 0 for block in blocks)
            prefix = f'{regime}.{control}'
            architecture.extend([condition(prefix+'.success', shared['found'], other['found'], shared['found'] >= other['found']),
                                 condition(prefix+'.moves', shared['steps'], .95*other['steps'], shared['steps'] <= .95*other['steps']),
                                 condition(prefix+'.positive_blocks', wins, 6, wins >= 6),
                                 condition(prefix+'.cost', shared['controller_seconds'], other['controller_seconds'], shared['controller_seconds'] <= other['controller_seconds'])])
        regimes[regime] = {'weights': {str(h): w for h, w in weight.items()}, 'means': means,
                          'family_means': family, 'blocks': blocks,
                          'strata': {str(h): {a: {m: math.fsum(float(r[m]) for r in subset if r['arm'] == a and r['initial_hit'] == h)/8
                                                 for m in METRICS} for a in ARMS} for h in (1, 2, 3)},
                          'raw_counts': {a: {'found': sum(r['found'] for r in subset if r['arm'] == a), 'episodes': 24} for a in ARMS}}
    require((len(competence), len(compression), len(architecture)) == (18, 12, 24), 'complete54 fixed scientific checks')
    return {'episodes': 720, 'regimes': regimes, 'competence_checks': competence, 'compression_checks': compression,
            'architecture_checks': architecture, 'pilot_continuation': all(c['passes'] for c in competence+compression+architecture),
            'learned_architecture_advantage_established': False, 'inherited_gate_revised': False}


RUNNER = 'scripts/study_otto_symmetry_head.py'
SCOPE = ('Independent saved-only public filter/features, fixed sampled rows and pooled exposure, complete checkpoints '
         'and reported optimizer-step/order witnesses, local NumPy readouts, selected actions, cohort/cost arithmetic '
         'and54 conditions. No training or simulator/remote calls. Original optimizer trajectories, teacher analytic '
         'cost generation, native kernel/RNG execution, Torch export parity and timing truth remain authenticated '
         'producer evidence. Local saved-checkpoint computations are counted, not described as zero model computation.')


def payload_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'collection-transitions.jsonl', 'collection-episodes.jsonl', 'work-contexts.jsonl', 'work.jsonl',
             'fits.jsonl', 'fit-curves.jsonl', 'epoch-orders.jsonl', 'inference-setup.json',
             'eval-transitions.jsonl', 'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json',
             'pooled-weights.npz', 'pooling.json', 'training-costs.json'}
    names.update(f'kernel-{r}.npz' for r in REGIMES)
    for split in ('train', 'valid', 'dagger'):
        names.update((f'{split}-data.npz', f'{split}-rows.jsonl'))
    names.update(f'{phase}-{family}-{seed}.npz' for phase in ('initial', 'final') for seed in FIT_SEEDS for family in ('shared', 'dense'))
    return names


def saved_choice(costs, allowed, neural, np):
    require(len(costs) == 4 and allowed == sorted(set(allowed)) and allowed, 'all four costs and canonical eligible IDs')
    if neural:
        require(all(type(v) in (int, float) and math.isfinite(v) for v in costs), 'all4 finite raw learned costs')
        values = np.asarray(costs, np.float32)
        require(values.tolist() == costs, 'saved costs exactly represent float32')
    else:
        require(all((type(costs[a]) in (int, float) and math.isfinite(costs[a])) if a in allowed else costs[a] is None for a in range(4)), 'analytic null outside eligible set')
        values = np.asarray([math.inf if v is None else v for v in costs], np.float64)
    permitted = values[allowed]
    return allowed[int(np.flatnonzero(np.abs(permitted-permitted.min()) < 1e-10)[0])]


def posterior_record(probability):
    return {'sha256': hashlib.sha256(probability.tobytes()).hexdigest(), 'mass': float(probability.sum())}


class Work:
    """Stream complete ordered calls; never materialize millions of events."""
    def __init__(self, iterator, contexts, compare):
        self.iterator, self.contexts, self.c = iterator, contexts, compare
        self.counts, self.sequence = {}, 0
        self.used_contexts = set()

    def call(self, channel, context):
        attempted = next(self.iterator)
        self.sequence += 1
        self.c.equal(attempted[:3], [self.sequence, 0, channel], 'ordered durable attempt')
        require(len(attempted) == 5, 'compact work attempt schema')
        require(type(attempted[3]) is int and 0 <= attempted[3] < len(self.contexts), 'valid context index')
        self.used_contexts.add(attempted[3])
        reconstructed = dict(self.contexts[attempted[3]])
        if attempted[4] is not None:
            reconstructed['step'] = attempted[4]
        self.c.equal(reconstructed, context, 'causal work context')
        returned = next(self.iterator)
        self.c.equal(returned[:3], [self.sequence, 1, channel], 'matched durable return')
        require(len(returned) == 6 and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in returned[3:]), 'finite recorded operation intervals')
        seconds, raw, excluded = returned[3:]
        self.c.close(seconds, raw-excluded, 'net recorded interval')
        counts = self.counts.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        counts['attempted'] += 1
        counts['returned'] += 1
        counts['seconds'] += seconds
        return {'seconds': seconds, 'instrumented_seconds': raw, 'excluded_io_seconds': excluded}


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
        require(rss <= LIMITS['rss_bytes'], 'saved audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'saved audit output cap')

    def authenticate(self):
        a, c = self.args, self.c
        for path in (a.plan, a.run, a.terminal, a.output):
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink paths')
        for path, pin in ((a.plan, a.plan_sha256), (a.run/'receipt.json', a.receipt_sha256), (a.terminal, a.terminal_sha256)):
            c.equal(B.digest(path, self.check)['sha256'], pin, 'external evidence identity before decode')
        plan, worker, terminal = B.read(a.plan), B.read(a.run/'receipt.json'), B.read(a.terminal)
        for name in (RUNNER, 'scripts/audit_otto_symmetry_head.py', 'tests/test_audit_otto_symmetry_head.py'):
            c.equal(B.digest(ROOT/name, self.check)['sha256'], plan['sources'][name], 'source before lineage import')
        reader = B.load(ROOT/RUNNER, '_symmetry_lineage_only')
        c.equal(reader.authenticate(a), plan, 'qualified stdlib producer lineage only')
        require(worker['status'] == 'completed' and worker['version'] == 'otto-symmetry-head-v1'
                and worker['completed_stage_fits'] == 12 and worker['completed_episodes'] == 720
                and worker['external_model_calls'] == 0 and worker['pending'] == [], 'complete successful run before scientific payloads')
        for key in ('sources', 'inputs', 'limits'):
            c.equal(worker[key], plan[key], 'worker frozen plan join')
        c.equal(worker['plan_sha256'], a.plan_sha256, 'worker external plan hash')
        c.equal(set(worker['files']), payload_names(), 'exact scientific payload manifest')
        c.equal({p.name for p in a.run.iterdir()}, payload_names() | {'receipt.json'}, 'no missing/late/extra payloads')
        for name, expected in worker['files'].items():
            path = a.run/name
            require(path.is_file() and not path.is_symlink(), 'regular authenticated payload')
            c.equal(B.digest(path, self.check), expected, 'full scientific payload hashes before arrays')
        started = B.read(a.run/'started.json')
        request, launch = started['request'], started['launch']
        c.equal(request, {'plan': str(a.plan), 'plan_sha256': a.plan_sha256, 'supervision': request['supervision'], 'output': str(a.run)}, 'actual worker arguments')
        c.equal(B.digest(Path(request['supervision']), self.check)['sha256'], worker['supervision_sha256'], 'actual launch pin')
        c.equal(B.read(Path(request['supervision'])), launch, 'worker launch contents')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['group_absent'] is True
                and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful absent supervisor before outcome decode')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend', 'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            c.equal(terminal[key], launch[key], 'supervisor launch identity')
        command = list(launch['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        c.equal(command[:2], [plan['python_executable'], str(ROOT/RUNNER)], 'actual interpreter/source')
        require(len(command) == 10, 'four CLI bindings')
        c.equal(dict(zip(command[2::2], command[3::2], strict=True)), {f'--{k.replace("_", "-")}': v for k, v in request.items()}, 'actual command arguments')
        require(launch['cap_seconds'] == 5400 and launch['pid'] == launch['pgid'] and launch['parent_pid'] != launch['pid']
                and launch['cwd'] == str(ROOT) and launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and worker['clock_backend'] == launch['clock_backend'] and launch['deadline_ns'] == launch['started_ns']+5400*10**9
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns'], 'strict native time enclosure')
        c.equal(started['started_ns'], worker['started_ns'], 'same worker origin')
        c.equal(worker['wall_seconds'], (worker['finished_ns']-worker['started_ns'])/1e9, 'worker native interval')
        c.equal(terminal['elapsed_ns'], terminal['finished_ns']-terminal['started_ns'], 'parent native interval')
        c.equal(terminal['wall_seconds'], terminal['elapsed_ns']/1e9, 'parent seconds')
        c.equal(launch['clock_source_sha256'], B.CLOCK_PIN, 'qualified clock identity')
        c.equal(launch['watchdog_sha256'], plan['sources']['scripts/supervise_dialogue_observation_v2.py'], 'qualified parent source')
        require(0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in a.run.iterdir()) <= plan['limits']['output_bytes'], 'recorded producer resource bounds')
        self.receipt.update(plan_sha256=a.plan_sha256, worker_sha256=a.receipt_sha256, terminal_sha256=a.terminal_sha256,
                            producer_source_sha256=plan['sources'][RUNNER], authentication_reuse='Producer.authenticate lineage only, no producer arithmetic.')
        return plan, worker

    def numerical_inputs(self):
        import numpy as np
        self.np = np
        self.kernels, self.mixtures, self.datasets, self.heads = {}, {}, {}, {}
        for regime in REGIMES:
            with np.load(self.args.run/f'kernel-{regime}.npz', allow_pickle=False) as z:
                self.c.equal(set(z.files), {'likelihood', 'initial_hit_weights'}, 'public kernel schema')
                k, w = z['likelihood'], z['initial_hit_weights']
            require(k.shape == (4, 107, 107) and k.dtype == np.float64 and np.isfinite(k).all()
                    and (k >= 0).all() and (k <= 1).all() and not k[:,53,53].any(), 'public kernel geometry')
            require(w.shape == (4,) and w[0] == 0 and np.isfinite(w).all() and (w[1:] > 0).all()
                    and abs(float(w.sum())-1) <= 1e-12, 'positive initial mixture')
            self.kernels[regime], self.mixtures[regime] = k, {h: float(w[h]) for h in (1,2,3)}
        for split, maxrows in (('train',12288), ('valid',3072), ('dagger',9216)):
            with np.load(self.args.run/f'{split}-data.npz', allow_pickle=False) as z:
                self.c.equal(set(z.files), {'features','target','valid','weights'}, 'four fixed data arrays')
                data = {k:z[k] for k in z.files}
            n = len(data['features'])
            require(0 < n <= maxrows, 'fixed collection row cap')
            for name,shape,dtype in (('features',(n,DIMENSION),np.float32),('target',(n,4),np.float32),('valid',(n,4),np.bool_),('weights',(n,),np.float32)):
                require(data[name].shape == shape and data[name].dtype == dtype and np.isfinite(data[name]).all(), 'fixed finite array shapes')
            data['rows'] = list(B.lines(self.args.run/f'{split}-rows.jsonl', self.check))
            self.c.equal([r['row_index'] for r in data['rows']], list(range(n)), 'complete canonical retained row indices')
            expected_weights = balanced_weights([r['episode_id'] for r in data['rows']], np)
            require(np.array_equal(data['weights'],expected_weights), 'episode-balanced weights before training')
            self.datasets[split] = data
        for phase in ('initial','final'):
            for seed in FIT_SEEDS:
                for family,kind in (('shared','d4_shared'),('dense','dense_augmented')):
                    name = f'{phase}-{family}-{seed}.npz'
                    with np.load(self.args.run/name, allow_pickle=False) as z:
                        self.c.equal(set(z.files), {'version','kind','input_dim',*WEIGHT_NAMES}, 'exact portable checkpoint arrays')
                        self.c.equal((z['version'].item(),z['kind'].item(),z['input_dim'].item()), ('otto-symmetry-head-v1',kind,DIMENSION), 'checkpoint architecture identity')
                        weights = {key:z[key] for key in WEIGHT_NAMES}
                    validate_weights(weights,kind,np)
                    self.heads[phase,f'{family}@{seed}'] = weights
        self.check()

    def predicted(self, feature, phase, arm, stored, allowed):
        family,seed = arm.split('@')
        key = f'{"dense" if family == "dense_ensemble" else family}@{seed}'
        mode = {'shared':'d4_shared','dense':'dense_augmented','dense_ensemble':'dense_ensemble'}[family]
        expected = readout(feature,self.heads[phase,key],mode,self.np)
        self.receipt['saved_checkpoint_readout_calls'] += 1
        self.receipt['saved_network_rows'] += 1 if family == 'dense' else 8
        actual = self.np.asarray(stored,self.np.float32)
        error = self.np.abs(actual.astype(self.np.float64)-expected.astype(self.np.float64))
        require(actual.shape == expected.shape == (4,) and self.np.isfinite(actual).all()
                and (error <= 2e-5+2e-5*self.np.abs(expected.astype(self.np.float64))).all(), 'independent checkpoint cost agreement')
        self.max_prediction_error = max(self.max_prediction_error,float(error.max()))
        if saved_choice(expected.tolist(),allowed,True,self.np) != saved_choice(stored,allowed,True,self.np):
            self.near_tie_choices += 1

    def qualification(self):
        np,c = self.np,self.c
        for _ in range(3):
            self.work.call('native_reset',{'phase':'setup'})
        saved = B.read(self.args.run/'qualification.json')
        transitions = iter(B.lines(self.args.run/'qualification.jsonl',self.check))
        rebuilt,total = [],0
        for case in range(8):
            context={'phase':'qualification','case':case}
            self.work.call('native_reset',context)
            reset=next(transitions)
            public=B.packet([26,26],1+case%3,False,0)
            c.equal(reset['kind'],'reset','qualified reset')
            c.equal(reset['case'],case,'qualified fixed case')
            c.equal(reset['public'],public,'qualified conditioned initial hit')
            source=reset['source_evaluation_only']
            probability=B.posterior(np.ones((53,53),np.float64)/2808,public,self.kernels['lambda5'],np)
            for step in range(1,33):
                self.work.call('native_step',{**context,'step':step})
                event=next(transitions);action=(0,2,1,3)[(step-1)%4]
                target=B.move(public['position'],action);found=target==source
                c.equal((event['kind'],event['case'],event['step'],event['action']),('step',case,step,action),'qualified prescribed action')
                after=B.packet(target,-2 if found else event['public']['hit'],found,step)
                c.equal(event['public'],after,'qualified public transition')
                probability=B.posterior(probability,after,self.kernels['lambda5'],np)
                c.equal(event['posterior'],posterior_record(probability),'qualified independent public filter')
                public=after;total+=1
                if found:
                    break
            rebuilt.append({'case':case,'steps':step,'found':found,'passed':True})
        B.exhausted(transitions,'mechanical qualification transitions')
        c.equal(saved['checks'],rebuilt,'all8 complete qualification cases')
        c.equal((saved['status'],saved['resets'],saved['native_steps']),('completed',8,total),'qualification bounded work')
        return total

    def episode(self, row, stage, identity, transitions):
        np,c=self.np,self.c
        regime,seed,hit,arm,block=identity
        episode_id=f'{stage}:{regime}:{seed}:{arm}'
        expected={'episode_id':episode_id,'stage':stage,'regime':regime,'seed':seed,'initial_hit':hit,'arm':arm,'block':block}
        c.equal({k:row[k] for k in expected},expected,'complete disjoint episode identity')
        steps=row['steps']
        require(type(steps) is int and 1 <= steps <= HORIZON and type(row['found']) is bool
                and (row['found'] or steps==HORIZON),'complete found/censored trajectory')
        context={'phase':stage,'episode':episode_id,'step':0}
        self.work.call('native_reset',context)
        public=B.packet([26,26],hit,False,0)
        source=row['source_evaluation_only']
        require(len(source)==2 and all(type(v) is int and 0 <= v < 53 for v in source) and source!=[26,26],'sampled evaluator source')
        probability=B.posterior(np.ones((53,53),np.float64)/2808,public,self.kernels[regime],np)
        state=posterior_record(probability)
        c.equal(next(transitions),{'kind':'reset',**expected,'public':public,'posterior_after':state,'source_evaluation_only':source},'initial public/source join')
        pair=(stage,regime,seed)
        if pair in self.sources:
            c.equal((source,public),self.sources[pair],'paired source and initial public packet')
        else:
            self.sources[pair]=(source,public)
        draws=iter(row['draws_evaluation_only'])
        first=next(draws)
        c.equal((first['channel'],first['index'],first['selected_index']),('source',0,53*source[0]+source[1]),'source draw index')
        retained=set(selected_prefixes(steps)) if stage!='eval' else set()
        neural='@' in arm
        totals={key:[] for key in ('choose_seconds','update_seconds','environment_seconds','choose_instrumented_seconds','choose_excluded_io_seconds')}
        for step in range(1,steps+1):
            self.check()
            ctx={**context,'step':step}
            teacher=self.work.call('teacher_label',ctx) if stage!='eval' else None
            operation=self.work.call('head_predict',ctx) if neural else (self.work.call('analytic_choose',ctx) if stage=='eval' else teacher)
            native=self.work.call('native_step',ctx)
            event=next(transitions)
            c.equal((event['kind'],event['episode_id'],event['step']),('step',episode_id,step),'ordered complete public decisions')
            allowed=public['valid_actions']
            c.equal(event['allowed_actions'],allowed,'public inbounds mask')
            action=saved_choice(event['costs'],allowed,neural,np)
            c.equal(event['action'],action,'exact first-index recorded-cost action')
            feature=feature_vector(probability,public,self.kernels[regime],REGIMES[regime],np) if neural or step-1 in retained else None
            if neural:
                self.predicted(feature,'final' if stage=='eval' else 'initial',arm,event['costs'],allowed)
            if step-1 in retained:
                data=self.datasets[stage];index=self.offsets[stage];record=data['rows'][index]
                canonical={'row_index':index,'episode_id':episode_id,'stage':stage,'regime':regime,'seed':seed,
                           'initial_hit':hit,'arm':arm,'prefix_index':step-1,'public':public,'posterior':state,
                           'teacher_costs':record['teacher_costs']}
                c.equal(record,canonical,'fixed sampled pre-action prefix and public posterior')
                if not neural:
                    c.equal(record['teacher_costs'],event['costs'],'teacher behavior/target identity')
                valid=np.asarray([a in allowed for a in range(4)],np.bool_)
                require(np.array_equal(feature,data['features'][index]) and np.array_equal(valid,data['valid'][index])
                        and np.array_equal(teacher_target(record['teacher_costs'],valid.tolist(),np),data['target'][index]),'independent saved features/relative target/mask')
                self.offsets[stage]+=1
            target=B.move(public['position'],action);found=target==source
            require(target!=public['position'] and (not found or step==steps),'moving action and first-found stopping')
            public_hit=event['public']['hit']
            require(type(public_hit) is int and (public_hit==-2 if found else 0 <= public_hit < 4),'observed hit or terminal sentinel')
            after=B.packet(target,public_hit,found,step)
            c.equal(event['public'],after,'movement and sampled found semantics')
            c.equal(event['native_p_end'],float(found),'native sampled termination')
            c.equal(event['posterior_before'],state,'pre-action public filter')
            probability=B.posterior(probability,after,self.kernels[regime],np)
            state=posterior_record(probability)
            c.equal(event['posterior_after'],state,'complete independently reconstructed update')
            if not found:
                draw=next(draws)
                c.equal((draw['channel'],draw['index'],draw['selected_index']),('hit',step-1,public_hit),'chronological hit draw')
            for key,values in totals.items():
                value=event[key]
                require(type(value) in (int,float) and math.isfinite(value) and value>=0,'nonnegative public-operation time')
                values.append(value)
            c.close(event['environment_seconds'],native['seconds'],'native work/transition duration join')
            c.close(event['choose_seconds'],event['choose_instrumented_seconds']-event['choose_excluded_io_seconds'],'choice excludes only recorded nested I/O')
            require(event['choose_seconds']+1e-9 >= operation['seconds']+(teacher['seconds'] if neural and teacher else 0),'readout/teacher inside complete choice')
            public=after
        B.exhausted(draws,'episode random draws')
        for draw in row['draws_evaluation_only']:
            require(0 <= draw['uniform'] < 1 and math.isfinite(draw['cdf_mass']) and abs(draw['cdf_mass']-1) < 1e-10,'valid recorded draw')
            key=(*pair,draw['channel'],draw['index'])
            if key in self.uniforms:
                c.equal(draw['uniform'],self.uniforms[key],'paired random-channel uniforms')
            else:
                self.uniforms[key]=draw['uniform']
        c.equal(row['final_public'],public,'complete final public observation')
        c.equal((row['found'],row['updates'],row['blocked_steps'],row['final_update_assimilated']), (public['done'],steps,0,True),'all final updates and outcomes retained')
        for key,values in totals.items():
            c.close(row[key],math.fsum(values),'summed complete operation costs')
        c.close(row['setup_allocation_seconds'],0.,'physical episode excludes later setup allocation')
        c.close(row['controller_seconds'],math.fsum(row[k] for k in TIMES),'physical complete controller cost')
        require(math.isfinite(row['init_seconds']) and row['init_seconds'] >= 0, 'finite actor initialization time')
        c.equal(row['state_bytes'],22472,'full exact public posterior storage')
        storage=row['storage']
        c.equal(set(storage), {'public_actor','feature_kernel_bytes','head'}, 'complete storage components')
        public_storage=storage['public_actor']
        c.equal({k:v for k,v in public_storage.items() if k!='scope'},
                {'immutable_array_bytes':366368+91592, 'mutable_array_bytes':22472,
                 'immutable_arrays':{'observation_kernel':366368,'manhattan_distance_table':91592}}, 'public actor array storage')
        c.equal(storage['feature_kernel_bytes'],366368 if neural or stage!='eval' else 0,'feature kernel allocation')
        c.equal({k:v for k,v in storage['head'].items() if k!='scope'} if neural else storage['head'],
                head_storage(arm.split('@')[0]) if neural else None,'retained runtime head and shared D4 tables')
        self.episode_counts[stage]+=1
        return row

    def fit_stage(self, stage, fits, curves, orders):
        np,c = self.np,self.c
        parts = ('train',) if stage == 'initial' else ('train','dagger')
        records = [{k:v for k,v in r.items() if k != 'row_index'} for part in parts for r in self.datasets[part]['rows']]
        arrays = {k: np.concatenate([self.datasets[p][k] for p in parts]) for k in ('features','target','valid')}
        arrays['weights'] = balanced_weights([r['episode_id'] for r in records], np)
        hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        row_hash = hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        rows,episodes = len(records),len({r['episode_id'] for r in records})
        require(episodes == (192 if stage == 'initial' else 336), 'all collection episodes in common fit data')
        delta = 40*math.ceil(rows/128)
        if stage == 'final':
            with np.load(self.args.run/'pooled-weights.npz',allow_pickle=False) as z:
                c.equal(z.files,['weights'],'pooled weight closure')
                require(np.array_equal(z['weights'],arrays['weights']),'shared recomputed pooled episode weights')
            c.equal(B.read(self.args.run/'pooling.json'), {'order':['train','dagger'],'rows':rows,'episodes':episodes,
                'shared_for_all_six_final_fits':True,'datasets':{p:B.digest(self.args.run/f'{p}-data.npz') for p in parts},
                'weights_sha256':hashes['weights']},'identical pooled exposure for all six final fits')
        for seed in FIT_SEEDS:
            for family in ('shared','dense'):
                fit_id=f'{family}@{seed}'
                row=next(fits)
                expected={'stage':stage,'fit_id':fit_id,'family':family,'seed':seed,'epochs':40,
                          'training_rows':rows,'training_episodes':episodes,'row_order_sha256':row_hash,
                          'row_weights_sha256':hashes['weights'],'training_array_sha256':hashes,
                          'optimizer_continued':stage=='final'}
                c.equal({k:row[k] for k in expected},expected,'all twelve fits share the exact declared data')
                before=[] if stage=='initial' else self.optimizer_steps[fit_id]
                after=[(before[0] if before else 0)+delta]*6
                c.equal(row['optimizer_steps_before'],before,'Adam step witness begins at prior final state')
                c.equal(row['optimizer_steps_after'],after,'all six parameter state steps advance completely')
                self.optimizer_steps[fit_id]=after
                rng=np.random.default_rng(seed+(0 if stage=='initial' else 100000))
                paid=[]
                for epoch in range(1,41):
                    self.check()
                    order=rng.permutation(rows)
                    c.equal(next(orders),{'stage':stage,'fit_id':fit_id,'epoch':epoch,'rows':rows,
                        'sha256':hashlib.sha256(order.tobytes()).hexdigest()},'paired independently reconstructed epoch order')
                    for _ in range(math.ceil(rows/128)):
                        paid.append(self.work.call('optimizer_update',{'phase':'fit','stage':stage,'fit_id':fit_id})['seconds'])
                curve=[]
                for epoch in range(5,41,5):
                    value=next(curves)
                    c.equal({k:value[k] for k in ('stage','fit_id','epoch')},{'stage':stage,'fit_id':fit_id,'epoch':epoch},'fixed eight descriptive validations')
                    require(set(value)=={'stage','fit_id','epoch','training_weighted_ce','weighted_ce','argmax_agreement'}
                            and all(type(value[k]) in (float,int) and math.isfinite(value[k]) and value[k]>=0
                                    for k in ('training_weighted_ce','weighted_ce','argmax_agreement'))
                            and value['argmax_agreement']<=1,'finite descriptive validation record')
                    curve.append(value)
                c.equal(row['curve'],curve,'all validation records retained with fixed final checkpoint')
                filename=f'{stage}-{family}-{seed}.npz'
                c.equal((row['checkpoint'],row['checkpoint_sha256']),(filename,B.digest(self.args.run/filename)['sha256']),'initial/final checkpoint identity')
                c.equal((row['export_validation_rows'],row['export_atol'],row['export_rtol'],row['export_action_identity_asserted']),
                        (list(range(min(16,len(self.datasets['valid']['rows'])))),2e-5,2e-5,False),'fixed export check witness, no invented exact-action qualification')
                require(math.isfinite(row['fit_seconds']) and row['fit_seconds']>=math.fsum(paid)-1e-8
                        and math.isfinite(row['export_max_abs_error']) and row['export_max_abs_error']>=0,'paid updates enclosed by each fit')
                self.fit_costs[stage]+=row['fit_seconds']
        self.stage_updates[stage]=delta*6

    def compute(self, plan, worker):
        self.numerical_inputs()
        c=self.c
        runtime=B.read(self.args.run/'runtime.json')
        c.equal(runtime['executable'],plan['python_executable'],'bound runtime interpreter')
        c.equal(runtime['versions'],plan['runtime_versions'],'bound numerical versions')
        c.equal((runtime['torch_threads'],runtime['torch_interop_threads'],runtime['module_allocation_episodes']),(1,1,648),'single thread and all learned setup recipients')
        c.equal(runtime['environment'],{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},'fixed CPU thread limits')
        for field in ('shared_setup_seconds','model_module_setup_seconds'):
            require(math.isfinite(runtime[field]) and runtime[field]>=0,'finite disjoint setup durations')
        contexts=list(B.lines(self.args.run/'work-contexts.jsonl',self.check))
        c.equal([r['id'] for r in contexts],list(range(len(contexts))),'contiguous work context identities')
        require(all(set(r)=={'id','context'} and 'step' not in r['context'] for r in contexts)
                and len({json.dumps(r['context'],sort_keys=True) for r in contexts})==len(contexts),'unique exact phase/episode contexts')
        self.work=Work(iter(B.lines(self.args.run/'work.jsonl',self.check)),[r['context'] for r in contexts],c)
        self.sources,self.uniforms,self.optimizer_steps={},{},{}
        self.offsets={p:0 for p in self.datasets}
        self.episode_counts={p:0 for p in ('train','valid','dagger','eval')}
        self.fit_costs={'initial':0.,'final':0.}
        self.stage_updates={}
        self.max_prediction_error,self.near_tie_choices=0.,0
        setup=B.read(self.args.run/'native-setup.json')
        c.equal(set(setup),{'checks','native_resets'},'native template witness closure')
        c.equal(setup['native_resets'],3,'three public model templates')
        c.equal([r['regime'] for r in setup['checks']],list(REGIMES),'complete public regimes')
        for index,row in enumerate(setup['checks']):
            c.equal((row['template_seed'],row['old_kernel_exact']),(1000101+index,index<2),'prior kernel witnesses and fixed template seeds')
            require(math.isfinite(row['maximum_kernel_formula_error']) and 0<=row['maximum_kernel_formula_error']<=1e-12,'qualified public kernel formula witness')
        qualification_steps=self.qualification()
        episodes=iter(B.lines(self.args.run/'collection-episodes.jsonl',self.check))
        transitions=iter(B.lines(self.args.run/'collection-transitions.jsonl',self.check))
        fits=iter(B.lines(self.args.run/'fits.jsonl',self.check))
        curves=iter(B.lines(self.args.run/'fit-curves.jsonl',self.check))
        orders=iter(B.lines(self.args.run/'epoch-orders.jsonl',self.check))
        all_rows=[]
        for stage in ('train','valid','dagger'):
            for identity in episode_order(stage):
                row=self.episode(next(episodes),stage,identity,transitions)
                all_rows.append({k:v for k,v in row.items() if k!='draws_evaluation_only'})
            c.equal(self.offsets[stage],len(self.datasets[stage]['rows']),'no omitted/extra retained prefixes')
            if stage=='valid':
                self.fit_stage('initial',fits,curves,orders)
        self.fit_stage('final',fits,curves,orders)
        for iterator,name in ((episodes,'collection episodes'),(transitions,'collection transitions'),(fits,'stage fits'),(curves,'validation curves'),(orders,'paired epoch orders')):
            B.exhausted(iterator,name)
        setup=B.read(self.args.run/'inference-setup.json')
        c.equal(set(setup),set(ARMS[:-1]),'nine separately loaded runtime heads')
        for arm,record in setup.items():
            family,seed=arm.split('@')
            family='dense' if family=='dense_ensemble' else family
            name=f'final-{family}-{seed}.npz'
            c.equal((record['checkpoint'],record['sha256'],record['allocated_episodes']),
                    (name,B.digest(self.args.run/name)['sha256'],72),'same dense weights for single and ensemble readouts')
            require(math.isfinite(record['seconds']) and record['seconds']>=0,'finite actual runtime head loading')
            c.equal({k:v for k,v in record['storage'].items() if k!='scope'},head_storage(family),'all retained immutable weights and common D4 arrays')
        physical=iter(B.lines(self.args.run/'eval-episodes.jsonl',self.check))
        allocated=iter(B.lines(self.args.run/'evaluation.jsonl',self.check))
        transitions=iter(B.lines(self.args.run/'eval-transitions.jsonl',self.check))
        rows=[]
        for identity in episode_order('eval'):
            row=self.episode(next(physical),'eval',identity,transitions)
            arm=row['arm']
            allocation=setup[arm]['seconds']/72+runtime['model_module_setup_seconds']/648 if arm in setup else 0.
            row={**row,'setup_allocation_seconds':allocation,'controller_seconds':row['controller_seconds']+allocation}
            c.equal(next(allocated),row,'complete cost with head/72 plus shared module/648 allocation')
            rows.append({k:v for k,v in row.items() if k!='draws_evaluation_only'})
        for iterator,name in ((physical,'physical evaluation'),(allocated,'allocated evaluation'),(transitions,'evaluation transitions'),(self.work.iterator,'durable work events')):
            B.exhausted(iterator,name)
        c.equal(self.episode_counts,{'train':192,'valid':48,'dagger':144,'eval':720},'all1104 autonomous episodes')
        require(self.work.used_contexts==set(range(len(contexts))),'all context records used')
        c.tree(worker['calls'],self.work.counts,'all attempted/returned operation counts and times')
        c.equal(worker['collection_episodes'],384,'all collection episodes complete')
        c.equal(worker['training_updates'],sum(self.stage_updates.values()),'all fixed optimizer updates')
        expected_calls={'native_reset':1115,'native_step':qualification_steps+sum(r['steps'] for r in all_rows+rows),
                        'teacher_label':sum(r['steps'] for r in all_rows),
                        'head_predict':sum(r['steps'] for r in all_rows+rows if '@' in r['arm']),
                        'analytic_choose':sum(r['steps'] for r in rows if r['arm']=='analytic_inbounds'),
                        'optimizer_update':sum(self.stage_updates.values())}
        c.equal({k:v['returned'] for k,v in self.work.counts.items()},expected_calls,'complete actual work coverage')
        for channel,limit in (('native_step','native_steps'),('native_reset','native_resets'),('optimizer_update','optimizer_updates')):
            require(expected_calls[channel]<=plan['limits'][limit],'frozen actual work caps')
        published=B.read(self.args.run/'summary.json')
        derived=aggregate(rows,self.mixtures)
        for key,value in derived.items():
            c.tree(published[key],value,'independent summary/'+key)
        c.equal(published['version'],'otto-symmetry-head-v1','published version')
        c.tree(published['calls'],self.work.counts,'summary actual operation work')
        c.equal(published['inference_setup'],setup,'complete actual runtime head setup')
        for name in ('shared_setup_seconds','model_module_setup_seconds'):
            c.equal(published[name],runtime[name],'recorded distinct setup components')
        c.equal(published['model_module_allocation_episodes'],648,'shared module cost counted once')
        c.equal(published['paired_source_cases'],72,'complete paired evaluation sources')
        c.equal(published['paired_uniforms'],sum(k[0]=='eval' for k in self.uniforms),'complete paired evaluation uniform indices')
        c.equal(published['all_public_beliefs_exact'],True,'complete exact public filter reconstruction')
        c.equal(worker['pilot_continuation'],derived['pilot_continuation'],'unchanged scientific decision')
        training=B.read(self.args.run/'training-costs.json')
        c.equal(set(training),{'train_collection_seconds','validation_collection_seconds','initial_fitting_seconds','dagger_collection_seconds','final_fitting_seconds'},'all disjoint collection/training phase costs')
        require(all(math.isfinite(v) and v>=0 for v in training.values()),'finite phase costs')
        for stage in ('initial','final'):
            require(training[f'{stage}_fitting_seconds']>=self.fit_costs[stage]-1e-8,'all fit durations enclosed in corresponding phase')
        for stage,key in (('train','train_collection_seconds'),('valid','validation_collection_seconds'),('dagger','dagger_collection_seconds')):
            paid=math.fsum(r['controller_seconds']+r['environment_seconds'] for r in all_rows if r['stage']==stage)
            require(training[key]>=paid-1e-8,'collection phase encloses all actor and environment work')
        evaluation_cost=math.fsum(r['controller_seconds']+r['environment_seconds'] for r in rows)
        disjoint=math.fsum(training.values())+evaluation_cost+runtime['shared_setup_seconds']
        require(disjoint<=worker['wall_seconds']+1e-8,'disjoint train/validation/evaluation/setup work within total worker interval')
        require(math.fsum(r['choose_excluded_io_seconds'] for r in all_rows+rows)<=worker['artifact_io_seconds']+1e-8,'measured exclusions bounded by all artifact I/O')
        return {'version':VERSION,'agreement':True,**derived,'comparisons':c.count,'maximum_scalar_difference':c.maximum_difference,
                'collection_episodes':384,'qualified_resets':11,'qualification_updates':qualification_steps,
                'public_updates':expected_calls['native_step'],'checkpoint_archives':12,'stage_fits':12,'epoch_orders':480,
                'descriptive_validation_records':96,'dataset_rows':dict(self.offsets),'all_episode_counts':self.episode_counts,
                'actual_work':expected_calls,'saved_checkpoint_readout_calls':self.receipt['saved_checkpoint_readout_calls'],
                'saved_network_rows':self.receipt['saved_network_rows'],'maximum_checkpoint_cost_difference':self.max_prediction_error,
                'recomputed_near_tie_choices_differing_within_tolerance':self.near_tie_choices,
                'condition_counts':{'competence':18,'compression':12,'architecture':24},
                'costs':{'training_collection_phases':training,'inference_setup':setup,
                         'shared_setup_seconds':runtime['shared_setup_seconds'],'model_module_setup_seconds':runtime['model_module_setup_seconds'],
                         'disjoint_accounted_seconds':disjoint,'worker_seconds':worker['wall_seconds']},
                'paired_cases':[{k:r[k] for k in ('regime','seed','arm','block','initial_hit','steps','found','controller_seconds')} for r in rows],
                'scope':SCOPE}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents),'exclusive regular absolute audit output')
        self.out.mkdir(parents=True,exist_ok=False)
        old_alarm=signal.getsignal(signal.SIGALRM)
        try:
            require(B.digest(ROOT/B.CLOCK)['sha256']==B.CLOCK_PIN,'qualified suspend-inclusive clock')
            self.clock=B.load(ROOT/B.CLOCK,'_symmetry_audit_clock').SuspendClock()
            self.start=self.clock.now_ns()
            signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('audit emergency wall cap')))
            signal.setitimer(signal.ITIMER_REAL,LIMITS['native_seconds'])
            self.receipt.update(source=B.digest(Path(__file__),self.check),independent_helper={'path':str(BASE_PATH),'sha256':BASE_PIN})
            B.write(self.out/'started.json',{'request':{k:str(v) for k,v in vars(self.args).items()},'limits':LIMITS,
                    'clock_backend':self.clock.backend,'started_ns':self.start})
            plan,worker=self.authenticate()
            result=self.compute(plan,worker)
            self.authenticate()
            B.write(self.out/'summary.json',result)
            self.check()
            finished=self.clock.now_ns()
            self.receipt.update(status='completed',agreement=True,clock_backend=self.clock.backend,started_ns=self.start,
                finished_ns=finished,wall_seconds=(finished-self.start)/1e9,comparisons=result['comparisons'],
                files={p.name:B.digest(p,self.check) for p in self.out.iterdir()})
            B.write(self.out/'receipt.json',self.receipt)
            self.check()
            return self.receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL,0)
            self.receipt.update(status='failed',agreement=False,error=repr(error),traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                B.write(self.out/'failed.json',self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Keep the primary failure.
                error.add_note(f'Failure preservation: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,old_alarm)


def head_storage(family):
    return {'parameter_array_bytes':(91329 if family=='shared' else 91380)*4,
            'shared_transform_array_bytes':8*DIMENSION*12+8*4*8+4*2*8,'mutable_array_bytes':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('plan','run','terminal','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    for flag in ('plan-sha256','receipt-sha256','terminal-sha256'):
        parser.add_argument('--'+flag,required=True)
    Audit(parser.parse_args()).execute()
