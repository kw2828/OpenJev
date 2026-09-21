"""Independent saved-capacity arithmetic; no optimizer or simulator execution.

Importing this file performs no input reads or model calls. Actual invocation
requires prospectively bound sources and completed external worker/parent pins.
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
from itertools import pairwise
from pathlib import Path

DIMENSION = 11028
SPATIAL_DIMENSION = 11025
SCALE = 64
WIDTHS = {'mlp8': (DIMENSION, 8, 1), 'mlp128': (DIMENSION, 128, 1),
          'deep128': (DIMENSION, 128, 128, 128, 1)}
SEEDS = (10101, 10102, 10103)
ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-capacity-saved-audit-v1'
RUNNER = 'scripts/study_otto_capacity.py'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'seconds': 600, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
SCOPE = ('Independent saved final dense readouts, all fixed TRAIN/VALID inputs and teacher-return targets, '
         'exact float32 alias groups, scalar error statistics, recorded optimization counts/orders and all12 conditions. '
         'No optimizer, simulator, action policy or new collection executes. Original cache posterior truth, '
         'Torch initialization/optimization/parity and timing truth remain authenticated producer evidence.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute regular nonsymlink input')
    return path


def digest(path, check=lambda: None):
    sha, size = hashlib.sha256(), 0
    with regular(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            check()
            sha.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': sha.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def lines(path, check=lambda: None):
    with Path(path).open() as stream:
        for line in stream:
            check()
            yield json.loads(line)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload_names():
    names = {'started.json', 'runtime.json', 'preparation.json', 'work.jsonl', 'initializations.jsonl',
             'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl', 'summary.json'}
    names.update(f'{phase}-{kind}-{seed}.npz' for seed in SEEDS for kind in WIDTHS
                 for phase in ('initial', 'final', 'predictions'))
    return names


def manifest(directory, pin, names, check):
    require(directory.is_absolute() and directory.is_dir()
            and not any(p.is_symlink() for p in (directory, *directory.parents)), 'regular evidence directory')
    require(digest(directory/'receipt.json', check)['sha256'] == pin, 'external completed receipt pin')
    receipt = read(directory/'receipt.json')
    require(receipt['status'] == 'completed' and set(receipt['files']) == set(names)
            and {p.name for p in directory.iterdir()} == set(names) | {'receipt.json'}, 'exact closed successful output')
    for name, expected in receipt['files'].items():
        require(Path(name).name == name and digest(directory/name, check) == expected, 'bound payload bytes before decoding')
    return receipt


def close(actual, expected, name, *, atol=1e-10, rtol=1e-10):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), name+' exact fields')
        for key in expected:
            close(actual[key], expected[key], name+'.'+key, atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name+' exact rows')
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close(left, right, name+f'[{i}]', atol=atol, rtol=rtol)
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), name+' numerical agreement')
    else:
        require(type(actual) is type(expected) and actual == expected, name+' exact identity')


def duplicate_floor(features, target, np, check=lambda: None):
    """Empirical row-weighted noise floor for byte-identical f32 inputs.

This is a property of the finite labeled cache, not a population noise estimate.
Byte identity intentionally retains signed-zero and exact training-cast details.
No labels, sample weights, or group memberships are changed for training.
"""
    require(features.ndim == 2 and features.dtype == np.float32 and len(features) > 0
            and features.shape[1] > 0 and np.isfinite(features).all(), 'finite nonempty float32 input matrix')
    require(target.shape == (len(features),) and target.dtype == np.float32
            and np.isfinite(target).all(), 'finite aligned float32 targets')
    groups = {}
    for index, row in enumerate(features):
        if index % 128 == 0:
            check()
        groups.setdefault(row.tobytes(order='C'), []).append(float(target[index]))
    squared_errors = []
    duplicate_groups = duplicate_rows = conflicting_groups = 0
    for index, values in enumerate(groups.values()):
        if index % 128 == 0:
            check()
        center = math.fsum(values) / len(values)
        squared_errors.append(math.fsum((value-center)**2 for value in values))
        if len(values) > 1:
            duplicate_groups += 1
            duplicate_rows += len(values)
        conflicting_groups += len(set(values)) > 1
    return {'rows': len(features), 'unique_groups': len(groups), 'duplicate_groups': duplicate_groups,
            'duplicate_rows': duplicate_rows, 'conflicting_groups': conflicting_groups,
            'irreducible_mse_normalized': math.fsum(squared_errors) / len(features)}


def scalar_metrics(predictions, target, np):
    """Independent summation, preserving signed predictions and physical units."""
    require(predictions.ndim == 1 and len(predictions) > 0 and predictions.dtype == np.float64
            and np.isfinite(predictions).all(), 'finite nonempty float64 predictions')
    require(target.shape == predictions.shape and target.dtype in (np.float32, np.float64)
            and np.isfinite(target).all(), 'finite aligned targets')
    differences = [float(p)-float(y) for p, y in zip(predictions, target, strict=True)]
    result = {'rows': len(predictions), 'mse_normalized': math.fsum(d*d for d in differences) / len(differences),
              'mae_physical': SCALE * math.fsum(abs(d) for d in differences) / len(differences),
              'negative_predictions': int(np.count_nonzero(predictions < 0)),
              'minimum_normalized': float(predictions.min()), 'maximum_normalized': float(predictions.max())}
    require(all(math.isfinite(value) for value in result.values()), 'finite scalar metrics')
    return result


def dense_predict(features, layers, c0, np):
    """Read stored parameters in output-by-input orientation with f64 arithmetic.

Every hidden layer has a bias and ReLU; the final layer has one linear output.
The supplied immutable float64 tensors are independent upcasts of stored f32
parameters. This helper never normalizes an input or clamps an output.
"""
    require(features.ndim == 2 and features.shape[1] == DIMENSION and len(features) > 0
            and features.dtype == np.float64 and np.isfinite(features).all(), 'finite float64 value inputs')
    require(len(layers) >= 2 and type(c0) in (float, int) and math.isfinite(c0), 'finite scalar baseline and hidden layer')
    hidden = features
    with np.errstate(over='raise', invalid='raise'):
        for index, (weight, bias) in enumerate(layers):
            require(weight.ndim == 2 and weight.shape[1] == hidden.shape[1]
                    and bias.shape == (weight.shape[0],) and weight.dtype == bias.dtype == np.float64
                    and np.isfinite(weight).all() and np.isfinite(bias).all(), 'aligned finite dense layer')
            hidden = hidden @ weight.T + bias
            if index + 1 < len(layers):
                hidden = np.maximum(hidden, 0)
        require(hidden.shape == (len(features), 1), 'one scalar output per input')
        result = c0 * features[:, :SPATIAL_DIMENSION].sum(axis=1, dtype=np.float64) + hidden[:, 0]
    require(np.isfinite(result).all(), 'finite signed deployed predictions')
    return result


def count_parameters(layers):
    return sum(weight.size+bias.size for weight, bias in layers)


def checkpoint(archive, kind, c0, np):
    """Validate the exact exported schema before independently upcasting it."""
    require(kind in WIDTHS, 'declared dense capacity family')
    dims = WIDTHS[kind]
    tensors = {f'{prefix}_{i}' for i in range(len(dims)-1) for prefix in ('weight', 'bias')}
    require(set(archive) == {'version', 'kind', 'input_dim', 'c0', *tensors}, 'exact checkpoint member closure')
    for key in ('version', 'kind', 'input_dim'):
        require(archive[key].shape == (), 'scalar checkpoint metadata')
    require(archive['version'].dtype.kind in 'US' and archive['version'].item() == 'otto-capacity-value-v1'
            and archive['kind'].dtype.kind in 'US' and archive['kind'].item() == kind
            and archive['input_dim'].dtype.kind in 'iu' and archive['input_dim'].item() == DIMENSION,
            'exact checkpoint architecture identity')
    require(archive['c0'].shape == () and archive['c0'].dtype == np.float32
            and np.isfinite(archive['c0']) and float(archive['c0']) == c0, 'same finite TRAIN-only float32 baseline')
    layers = []
    for i, (left, right) in enumerate(pairwise(dims)):
        weight, bias = archive[f'weight_{i}'], archive[f'bias_{i}']
        require(weight.shape == (right, left) and bias.shape == (right,)
                and weight.dtype == bias.dtype == np.float32
                and np.isfinite(weight).all() and np.isfinite(bias).all(), 'finite exact float32 layer shapes')
        layers.append((weight.astype(np.float64), bias.astype(np.float64)))
    return layers


def aggregate(fits, floor):
    require([row['fit_id'] for row in fits] == [f'{kind}@{seed}' for seed in SEEDS for kind in WIDTHS],
            'all nine final fits in seed-then-kind order')
    require(type(floor) in (int, float) and math.isfinite(floor) and floor >= 0, 'nonnegative finite alias floor')
    metrics = {row['fit_id']: row['metrics'] for row in fits}
    for panels in metrics.values():
        require(set(panels) == {'train', 'valid'}, 'both complete prediction splits')
        for split, n in (('train', 5589), ('valid', 1109)):
            panel = panels[split]
            require(panel['rows'] == n and type(panel['mse_normalized']) in (int, float)
                    and math.isfinite(panel['mse_normalized']) and panel['mse_normalized'] >= 0,
                    'complete nonnegative scalar error')
    checks = []
    for kind in ('mlp128', 'deep128'):
        for seed in SEEDS:
            candidate, narrow = metrics[f'{kind}@{seed}'], metrics[f'mlp8@{seed}']
            for name, value, threshold in (
                ('train_excess', candidate['train']['mse_normalized']-floor, .8*(narrow['train']['mse_normalized']-floor)),
                ('valid_mse', candidate['valid']['mse_normalized'], .9*narrow['valid']['mse_normalized']),
            ):
                checks.append({'name': f'{kind}.{seed}.{name}', 'value': value, 'threshold': threshold, 'passes': value <= threshold})
    return {'checks': checks,
            'family_admission': {kind: all(row['passes'] for row in checks if row['name'].startswith(kind+'.'))
                                 for kind in ('mlp128', 'deep128')},
            'capacity_screen_passed': all(row['passes'] for row in checks)}


def validate_cache(data, rows, split, np, check=lambda: None):
    """Rebuild features and teacher-return targets from the bound scalar cache.

The original completed independent audit authenticates the cached posteriors
against teacher transitions. This reader checks their bytes/metadata and
independently rebuilds every feature and target, without replaying a simulator.
"""
    require(split in ('train', 'valid') and set(data) == {'features', 'target', 'beliefs', 'positions', 'sensing_length'},
            'exact scalar cache closure')
    n = len(rows)
    require(n == {'train': 5589, 'valid': 1109}[split], 'all declared teacher prefixes')
    for name, shape, dtype in [('features', (n, DIMENSION), np.float32), ('target', (n,), np.float32),
                               ('beliefs', (n, 53, 53), np.float64), ('positions', (n, 2), np.int64),
                               ('sensing_length', (n,), np.float64)]:
        require(data[name].shape == shape and data[name].dtype == dtype and np.isfinite(data[name]).all(),
                'finite exact cache array geometry')
    episodes = []
    cursor = 0
    count = 96 if split == 'train' else 24
    firsts = (910001, 920001) if split == 'train' else (930001, 940001)
    keys = {'row_index', 'episode_id', 'stage', 'regime', 'seed', 'initial_hit', 'prefix_index',
            'total_steps', 'public', 'posterior', 'target'}
    for sensing, first in zip((3, 4), firsts, strict=True):
        for case in range(count):
            check()
            require(cursor < n, 'no missing teacher episode')
            steps = rows[cursor]['total_steps']
            require(type(steps) is int and 1 <= steps <= 2188, 'positive uncensored teacher duration')
            episode_id = f'{split}:lambda{sensing}:{first+case}:teacher'
            episodes.append(episode_id)
            for t in range(steps):
                require(cursor < n, 'no missing teacher prefix')
                row = rows[cursor]
                require(set(row) == keys and type(row['row_index']) is int and row['row_index'] == cursor
                        and (row['episode_id'], row['stage'], row['regime'], row['seed'], row['initial_hit'],
                             row['prefix_index'], row['total_steps'])
                        == (episode_id, split, f'lambda{sensing}', first+case, 1+case % 3, t, steps),
                        'every canonical teacher prefix in original order')
                p, position = data['beliefs'][cursor], data['positions'][cursor]
                require((p >= 0).all() and (position >= 0).all() and (position <= 52).all(), 'valid public probability and position')
                x, y = map(int, position)
                public = row['public']
                allowed = [action for action, possible in enumerate((x > 0, x < 52, y > 0, y < 52)) if possible]
                require(set(public) == {'position', 'hit', 'done', 'step', 'valid_actions'}
                        and public['position'] == [x, y] and public['done'] is False
                        and type(public['step']) is int and public['step'] == t
                        and type(public['hit']) is int and 0 <= public['hit'] <= 3
                        and public['valid_actions'] == allowed, 'nonterminal public row contract')
                require(row['posterior']['sha256'] == hashlib.sha256(p.tobytes(order='C')).hexdigest()
                        and row['posterior']['mass'] == float(p.sum(dtype=np.float64)), 'cached posterior witness')
                field = np.zeros((105, 105), np.float64)
                field[52-x:105-x, 52-y:105-y] = p
                mass = field.sum(dtype=np.float64)
                feature = np.concatenate((field.ravel(), np.asarray([mass*x/52, mass*y/52, mass*sensing/5])))
                target = np.float32((steps-t)/64)
                require(np.array_equal(data['features'][cursor], feature.astype(np.float32))
                        and data['target'][cursor] == target and row['target'] == float(target)
                        and data['sensing_length'][cursor] == sensing, 'independent exact feature and target reconstruction')
                cursor += 1
    require(cursor == n, 'no extra or selected teacher prefixes')
    return {'rows': n, 'episodes': len(episodes), 'feature_sha256': hashlib.sha256(data['features'].tobytes()).hexdigest(),
            'target_sha256': hashlib.sha256(data['target'].tobytes()).hexdigest()}


class Work:
    """Consume the exact one-level attempt/return journal in prescribed order."""
    def __init__(self, rows):
        self.rows, self.sequence, self.counts = iter(rows), 0, {}

    def take(self, channel, context):
        self.sequence += 1
        expected = {'id': self.sequence, 'channel': channel, 'context': context}
        attempt, returned = next(self.rows), next(self.rows)
        require(type(attempt['id']) is int and type(returned['id']) is int
                and attempt == {**expected, 'event': 'attempt'}
                and set(returned) == {*expected, 'event', 'seconds'}
                and {k: returned[k] for k in expected} == expected and returned['event'] == 'return',
                'exact ordered attempt/return context without nested or missing work')
        seconds = returned['seconds']
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'finite nonnegative recorded cost')
        count = self.counts.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        count['attempted'] += 1
        count['returned'] += 1
        count['seconds'] += seconds
        return seconds

    def finish(self):
        require(next(self.rows, None) is None, 'no extra work calls')
        expected = {'model_initialization': 9, 'optimizer_initialization': 9, 'checkpoint_export': 18,
                    'optimizer_update': 31680, 'parity_restore': 9, 'parity_forward': 18, 'saved_prediction': 243}
        require({k: v['returned'] for k, v in self.counts.items()} == expected, 'exact full-study channel counts')
        return self.counts


def check_terminal(plan, plan_path, worker, started, terminal, run, check):
    launch, request = started['launch'], started['request']
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['errors'] == [], 'successful reaped parent without timeout or cleanup error')
    for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                'cap_seconds', 'watchdog_sha256', 'clock_source_sha256'):
        require(launch[key] == terminal[key], 'same supervised launch and terminal')
    require(request == {'plan': str(plan_path), 'plan_sha256': digest(plan_path, check)['sha256'],
                        'output': str(run), 'supervision': request['supervision']}, 'exact worker request')
    require(digest(Path(request['supervision']), check)['sha256'] == worker['supervision_sha256']
            and read(request['supervision']) == launch, 'exact external launch identity')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command == [plan['python_executable'], str(ROOT/RUNNER), '--plan', str(plan_path),
                        '--plan-sha256', request['plan_sha256'], '--output', str(run),
                        '--supervision', request['supervision']], 'exact supervised interpreter/command')
    require(launch['cwd'] == str(ROOT) and launch['pid'] == launch['pgid'] != launch['parent_pid']
            and all(type(launch[k]) is int and launch[k] > 0 for k in ('pid', 'pgid', 'parent_pid')),
            'isolated original process identity')
    require(all(type(x) is int and x >= 0 for x in (launch['started_ns'], worker['started_ns'],
            worker['finished_ns'], terminal['finished_ns'], launch['deadline_ns'])), 'integer native timing')
    require(worker['clock_backend'] == launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
            and launch['clock_source_sha256'] == CLOCK_PIN
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']
            and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
            and started['started_ns'] == worker['started_ns']
            and launch['cap_seconds'] == plan['limits']['native_seconds'] == 1800
            and launch['deadline_ns'] == launch['started_ns']+1800*10**9
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9, 'strict native parent/worker enclosure')


def epoch_records(fit_id, epoch, expected_order, order_row, update_rows, curve, work, np):
    order_sha = hashlib.sha256(expected_order.tobytes()).hexdigest()
    close(order_row, {'fit_id': fit_id, 'epoch': epoch, 'order': expected_order.tolist(), 'sha256': order_sha}, 'epoch permutation')
    weighted, seconds = [], 0.
    for batch, offset in enumerate(range(0, len(expected_order), 128)):
        n = min(128, len(expected_order)-offset)
        context = {'fit_id': fit_id, 'epoch': epoch, 'batch': batch}
        row = next(update_rows)
        require(set(row) == {*context, 'rows', 'loss', 'gradient_norm', 'update_index'}
                and {k: row[k] for k in context} == context and type(row['rows']) is int and row['rows'] == n
                and type(row['update_index']) is int and row['update_index'] == (epoch-1)*44+batch+1,
                'every atomic update including final short batch')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ('loss', 'gradient_norm')), 'finite recorded loss and preclip gradient norm')
        weighted.append(row['loss']*n)
        seconds += work.take('optimizer_update', context)
    close(curve, {'fit_id': fit_id, 'epoch': epoch, 'rows': len(expected_order), 'updates': len(weighted),
                  'training_mse_normalized': math.fsum(weighted)/len(expected_order), 'order_sha256': order_sha}, 'online epoch loss accounting')
    return seconds


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = None
        self.start = self.finished = None
        self.readout_calls = self.readout_rows = 0
        self.maximum_prediction_difference = 0.

    def check(self):
        require(self.clock.now_ns() < self.deadline, 'independent audit native deadline')
        self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        require(self.rss <= LIMITS['rss_bytes'], 'independent audit RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'independent audit output cap')

    def authenticate(self):
        args = self.args
        require(digest(args.plan, self.check)['sha256'] == args.plan_sha256, 'external prospective plan pin')
        plan = read(args.plan)
        require(plan['version'] == 'otto-capacity-v1' and plan['mode'] == 'study'
                and plan['status'] == 'frozen_before_execution' and plan['independent_audit_limits'] == LIMITS,
                'fixed full-study and independent-audit allocation')
        own = str(Path(__file__).resolve().relative_to(ROOT))
        require(plan['sources'][own] == digest(Path(__file__), self.check)['sha256']
                and 'tests/test_audit_otto_capacity.py' in plan['sources'] and plan['sources'][CLOCK] == CLOCK_PIN,
                'prospectively pinned independent source and tests')
        for name, pin in plan['sources'].items():
            require(digest(ROOT/name, self.check)['sha256'] == pin, 'complete unchanged source closure')
        require(digest(args.terminal, self.check)['sha256'] == args.terminal_sha256, 'external successful parent pin')
        worker = manifest(args.run, args.receipt_sha256, payload_names(), self.check)
        require(worker['version'] == 'otto-capacity-v1' and worker['mode'] == 'study'
                and worker['sources'] == plan['sources'] and worker['inputs'] == plan['inputs']
                and worker['limits'] == plan['limits'] and worker['plan_sha256'] == args.plan_sha256
                and worker['completed_fits'] == 9 and worker['pending'] == [] and worker['parity_passed'] is True
                and all(worker[k] == 0 for k in ('external_model_calls', 'native_steps', 'native_resets')),
                'complete original study before decoding numerical payloads')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in args.run.iterdir()) <= plan['limits']['output_bytes'], 'saved worker resource bounds')
        check_terminal(plan, args.plan, worker, read(args.run/'started.json'), read(args.terminal), args.run, self.check)
        # The sole producer reuse authenticates immutable lineage/runtime only.
        producer = load(ROOT/RUNNER, '_capacity_audit_lineage')
        require(producer.authenticate(argparse.Namespace(plan=args.plan, plan_sha256=args.plan_sha256)) == plan,
                'prior input/audit/qualification closures and runtime identities')
        runtime = read(args.run/'runtime.json')
        close(runtime, {'python': sys.version, 'executable': sys.executable,
                        'threads': {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                        'torch_threads': 1, 'torch_interop_threads': 1, 'torch_deterministic': True}, 'recorded worker runtime')
        self.plan, self.worker = plan, worker

    def arrays(self, path):
        self.check()
        with self.np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        self.check()
        return result

    def predict(self, x, layers, c0):
        self.check()
        self.readout_calls += 1
        result = dense_predict(x, layers, c0, self.np)
        self.readout_rows += len(x)
        self.check()
        return result

    def equal_array(self, saved, expected, name):
        np = self.np
        require(saved.shape == expected.shape and saved.dtype == np.float64 and np.isfinite(saved).all(), name+' finite exact shape/dtype')
        delta = np.abs(saved-expected)
        require(bool(np.all(delta <= 1e-10+1e-10*np.abs(expected))), name+' independent float64 agreement')
        self.maximum_prediction_difference = max(self.maximum_prediction_difference, float(delta.max(initial=0)))

    def compute(self):
        import numpy as np
        self.np = np
        run = self.args.run
        data, evidence = {}, {}
        for split in ('train', 'valid'):
            full = self.arrays(self.plan['inputs'][split+'_data']['path'])
            rows = list(lines(self.plan['inputs'][split+'_rows']['path'], self.check))
            evidence[split] = validate_cache(full, rows, split, np, self.check)
            data[split] = (full['features'], full['target'])
            del full, rows
        x, y = data['train']
        c0 = float(np.float32(np.mean(y, dtype=np.float64)))
        floor = duplicate_floor(x, y, np, self.check)
        close(read(run/'preparation.json'), {'row_indices': {k: list(range(len(v[0]))) for k, v in data.items()},
              'c0_float32': c0, 'alias_floor': floor,
              'features': {k: {'sha256': evidence[k]['feature_sha256'], 'rows': len(v[0])} for k, v in data.items()}}, 'preparation')
        streams = {name: iter(lines(run/name, self.check)) for name in ('initializations.jsonl', 'epoch-orders.jsonl',
                   'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl')}
        work = Work(lines(run/'work.jsonl', self.check))
        fits, initial_first = [], {}
        probe = np.concatenate([data['train'][0][:8].astype(np.float64), data['valid'][0][:8].astype(np.float64),
                                np.zeros((1, DIMENSION)), data['train'][0][:1].astype(np.float64)*.5])
        for seed in SEEDS:
            for kind in WIDTHS:
                self.check()
                fit_id, context = f'{kind}@{seed}', {'fit_id': f'{kind}@{seed}'}
                fit = next(streams['fits.jsonl'])
                require((fit['fit_id'], fit['kind'], fit['seed'], fit['rows'], fit['epochs']) == (fit_id, kind, seed, 5589, 80), 'fixed final fit identity')
                initial_path, final_path = run/f'initial-{kind}-{seed}.npz', run/f'final-{kind}-{seed}.npz'
                initial = checkpoint(self.arrays(initial_path), kind, c0, np)
                require(all(not np.count_nonzero(bias) for _, bias in initial), 'all fresh initial biases are zero')
                initial_first[(seed, kind)] = initial[0][0]
                parameter_count = count_parameters(initial)
                close(next(streams['initializations.jsonl']), {'fit_id': fit_id, 'kind': kind, 'seed': seed,
                      'initial_sha256': self.worker['files'][initial_path.name]['sha256'],
                      'parameter_count': parameter_count, 'c0_float32': c0}, 'initial checkpoint witness')
                fit_call_seconds = work.take('model_initialization', context)+work.take('checkpoint_export', context)
                fit_call_seconds += work.take('optimizer_initialization', context)
                rng = np.random.default_rng(seed+20000)
                for epoch in range(1, 81):
                    fit_call_seconds += epoch_records(fit_id, epoch, rng.permutation(5589),
                        next(streams['epoch-orders.jsonl']), streams['updates.jsonl'], next(streams['fit-curves.jsonl']), work, np)
                fit_call_seconds += work.take('checkpoint_export', context)
                layers = checkpoint(self.arrays(final_path), kind, c0, np)
                parameter_names = {f'{key}_{i}' for i in range(len(layers)) for key in ('weight', 'bias')}
                require(set(fit) == {'fit_id', 'kind', 'seed', 'rows', 'epochs', 'fit_seconds', 'parameter_count',
                        'optimizer_steps', 'initial_sha256', 'checkpoint_sha256', 'metrics'}
                        and fit['optimizer_steps'] == dict.fromkeys(parameter_names, 3520)
                        and fit['parameter_count'] == parameter_count == count_parameters(layers)
                        and fit['initial_sha256'] == self.worker['files'][initial_path.name]['sha256']
                        and fit['checkpoint_sha256'] == self.worker['files'][final_path.name]['sha256']
                        and type(fit['fit_seconds']) in (int, float) and math.isfinite(fit['fit_seconds'])
                        and fit['fit_seconds'] >= fit_call_seconds, 'optimizer steps/checkpoints/paid fit-time joins')
                work.take('parity_restore', context)
                work.take('parity_forward', {**context, 'backend': 'numpy'})
                work.take('parity_forward', {**context, 'backend': 'torch'})
                parity = next(streams['parity.jsonl'])
                require(set(parity) == {'fit_id', 'features_sha256', 'rows', 'numpy', 'torch', 'passed'}
                        and parity['fit_id'] == fit_id and parity['rows'] == 18 and parity['passed'] is True
                        and parity['features_sha256'] == hashlib.sha256(probe.tobytes()).hexdigest(), 'fixed scalar parity witnesses')
                independent = self.predict(probe, layers, c0)
                saved_numpy, saved_torch = np.asarray(parity['numpy'], np.float64), np.asarray(parity['torch'], np.float64)
                self.equal_array(saved_numpy, independent, 'saved NumPy parity')
                self.equal_array(saved_torch, independent, 'independent Torch double parity')
                require(bool(np.all(np.abs(saved_numpy-saved_torch) <= 1e-10+1e-10*np.abs(saved_torch))),
                        'unchanged producer scalar parity predicate')
                predictions = self.arrays(run/f'predictions-{kind}-{seed}.npz')
                require(set(predictions) == {'train', 'valid'}, 'both final saved splits without selection')
                stats = {}
                for split, (vx, vy) in data.items():
                    saved = predictions[split]
                    require(saved.shape == (len(vx),) and saved.dtype == np.float64 and np.isfinite(saved).all(), 'complete saved prediction vector')
                    for offset in range(0, len(vx), 256):
                        work.take('saved_prediction', {**context, 'split': split, 'offset': offset})
                        value = self.predict(vx[offset:offset+256].astype(np.float64), layers, c0)
                        self.equal_array(saved[offset:offset+256], value, 'all independent cached-input predictions')
                    # Independently reduce authenticated saved vectors after numerical replay.
                    stats[split] = scalar_metrics(saved, vy, np)
                close(fit['metrics'], stats, 'final TRAIN/VALID metrics')
                fits.append({**fit, 'metrics': stats})
                del initial, layers, predictions
        for seed in SEEDS:
            narrow, wide, deep = (initial_first[(seed, kind)] for kind in WIDTHS)
            require(np.array_equal(narrow, wide[:8]) and np.array_equal(wide, deep), 'paired eight-row first-matrix initialization')
        for stream in streams.values():
            require(next(stream, None) is None, 'no extra fit/order/update/parity rows')
        calls = work.finish()
        close(self.worker['calls'], calls, 'full attempted/returned work and costs')
        require(math.fsum(fit['fit_seconds'] for fit in fits) <= self.worker['wall_seconds'], 'fit wall bounded by actual worker elapsed')
        result = aggregate(fits, floor['irreducible_mse_normalized'])
        expected = {'version': 'otto-capacity-v1', 'mode': 'study', 'rows': {'train': 5589, 'valid': 1109},
                    'alias_floor': floor, 'metrics': {fit['fit_id']: fit['metrics'] for fit in fits}, **result,
                    'qualification_projection_seconds': None, 'training_costs': {fit['fit_id']: fit['fit_seconds'] for fit in fits},
                    'learned_architecture_advantage_established': False,
                    'scope': 'Fixed-data scalar fitting only; no policy, simulator, held-out competence or novelty claim.'}
        close(read(run/'summary.json'), expected, 'all summary numbers and exact twelve decisions')
        require(self.readout_calls == 252 and self.readout_rows == 60444, 'complete independent final and parity readout allocation')
        return {**expected, 'audit_version': VERSION, 'agreement': True, 'scope': SCOPE,
                'reconstructed_cache': evidence, 'recorded_calls': calls,
                'independent_readout_calls': self.readout_calls, 'independent_readout_rows': self.readout_rows,
                'maximum_prediction_difference': self.maximum_prediction_difference}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists()
                and not any(p.is_symlink() for p in self.out.parents), 'exclusive nonsymlink audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified native clock')
            self.clock = load(ROOT/CLOCK, '_capacity_audit_clock').SuspendClock()
            self.start = self.clock.now_ns()
            self.deadline = self.start+LIMITS['seconds']*10**9
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('audit wall alarm')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            write(self.out/'started.json', {'version': VERSION, 'request': {k: str(v) for k, v in vars(self.args).items()},
                  'source': digest(Path(__file__)), 'limits': LIMITS, 'clock_backend': self.clock.backend,
                  'started_ns': self.start, 'deadline_ns': self.deadline, 'scope': SCOPE})
            self.authenticate()
            summary = self.compute()
            self.authenticate()
            write(self.out/'summary.json', summary)
            self.check()
            self.finished = self.clock.now_ns()
            require(self.finished < self.deadline, 'strict final deadline')
            receipt = {'version': VERSION, 'status': 'completed', 'agreement': True, 'source': digest(Path(__file__), self.check),
                       'plan_sha256': self.args.plan_sha256, 'worker_sha256': self.args.receipt_sha256,
                       'terminal_sha256': self.args.terminal_sha256, 'producer_source_sha256': self.plan['sources'][RUNNER],
                       'limits': LIMITS, 'clock_backend': self.clock.backend, 'started_ns': self.start,
                       'finished_ns': self.finished, 'elapsed_ns': self.finished-self.start,
                       'wall_seconds': (self.finished-self.start)/1e9, 'peak_rss_bytes': self.rss,
                       'independent_readout_calls': self.readout_calls, 'independent_readout_rows': self.readout_rows,
                       'scope': SCOPE, 'files': {name: digest(self.out/name, self.check) for name in ('started.json', 'summary.json')}}
            write(self.out/'receipt.json', receipt)
            self.check()
            signal.setitimer(signal.ITIMER_REAL, 0)
            return receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            failure = {'version': VERSION, 'status': 'failed', 'error': repr(error), 'scope': SCOPE,
                       'independent_readout_attempts': self.readout_calls, 'independent_returned_rows': self.readout_rows,
                       'started_ns': self.start, 'finished_ns': None, 'elapsed_ns': None, 'wall_seconds': None}
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json', failure)
            except BaseException as secondary:  # noqa: BLE001 - Retain original failure.
                error.add_note(f'Failure publication: {secondary!r}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Audit(parser.parse_args()).execute()
