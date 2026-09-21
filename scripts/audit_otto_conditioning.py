"""Independent saved conditioning audit; no optimizer, policy or simulator.

Uses the pinned capacity auditor's independent cache, scalar-statistic and
journal helpers. New gain arithmetic and initial-pair checks are defined here;
producer reuse is limited to source/input/runtime authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/audit_otto_capacity.py'
HELPER_PIN = '68b6c39837c8bf58ab519bf0437a7e7ffb39c0cd66fcf2c26274786fbf3073b0'
if hashlib.sha256((ROOT/HELPER).read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned independent capacity helper')
_spec = importlib.util.spec_from_file_location('_conditioning_independent_helpers', ROOT/HELPER)
C = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = C
_spec.loader.exec_module(C)
require, digest, read, write, lines, load = C.require, C.digest, C.read, C.write, C.lines, C.load
manifest, close, validate_cache = C.manifest, C.close, C.validate_cache
duplicate_floor, scalar_metrics, epoch_records = C.duplicate_floor, C.scalar_metrics, C.epoch_records
DIMENSION, SPATIAL_DIMENSION = 11028, 11025
WIDTHS = {'gain1': (DIMENSION, 8, 1), 'gain53': (DIMENSION, 8, 1)}
GAINS = {'gain1': 1, 'gain53': 53}
SEEDS = (10101, 10102, 10103)
VERSION = 'otto-conditioning-saved-audit-v1'
RUNNER = 'scripts/study_otto_conditioning.py'
CLOCK, CLOCK_PIN = C.CLOCK, C.CLOCK_PIN
LIMITS = {'seconds': 600, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
SCOPE = ('Independent saved final scalar readouts, unchanged TRAIN/VALID features and targets, '
         'alias floor, exact initial gain compensation, recorded update orders/counts and six conditions. '
         'No optimization, policy action, simulator or collection executes. Native Torch initialization, '
         'optimization, recorded Torch parity and timing truth remain source-bound execution evidence. '
         'This is a numerical-conditioning screen, not architecture or control competence evidence.')


def payload_names():
    names = {'started.json', 'runtime.json', 'preparation.json', 'work.jsonl', 'initializations.jsonl',
             'initial-pairing.jsonl', 'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl',
             'fits.jsonl', 'parity.jsonl', 'summary.json'}
    names.update(f'{phase}-{kind}-{seed}.npz' for seed in SEEDS for kind in WIDTHS
                 for phase in ('initial', 'final', 'predictions'))
    return names


def checkpoint(archive, kind, c0, np):
    require(kind in GAINS and set(archive) == {'version', 'kind', 'input_dim', 'spatial_gain', 'c0',
            'weight_0', 'bias_0', 'weight_1', 'bias_1'}, 'exact conditioned checkpoint member closure')
    for key in ('version', 'kind', 'input_dim', 'spatial_gain'):
        require(archive[key].shape == (), 'scalar checkpoint metadata')
    require(archive['version'].dtype.kind in 'US' and archive['version'].item() == 'otto-conditioned-value-v1'
            and archive['kind'].dtype.kind in 'US' and archive['kind'].item() == kind
            and archive['input_dim'].dtype.kind in 'iu' and archive['input_dim'].item() == DIMENSION
            and archive['spatial_gain'].dtype.kind in 'iu' and archive['spatial_gain'].item() == GAINS[kind],
            'exact conditioning identity and gain')
    require(archive['c0'].shape == () and archive['c0'].dtype == np.float32
            and np.isfinite(archive['c0']) and float(archive['c0']) == c0, 'unchanged float32 TRAIN baseline')
    layers = []
    for i, shape in enumerate(((8, DIMENSION), (1, 8))):
        weight, bias = archive[f'weight_{i}'], archive[f'bias_{i}']
        require(weight.shape == shape and bias.shape == (shape[0],) and weight.dtype == bias.dtype == np.float32
                and np.isfinite(weight).all() and np.isfinite(bias).all(), 'finite float32 width-eight tensors')
        layers.append((weight.astype(np.float64), bias.astype(np.float64)))
    return layers


def conditioned_predict(features, layers, c0, kind, np):
    require(kind in GAINS and features.ndim == 2 and features.shape[1] == DIMENSION and len(features) > 0
            and features.dtype == np.float64 and np.isfinite(features).all(), 'finite raw float64 inputs and declared gain')
    require(len(layers) == 2 and type(c0) in (int, float) and math.isfinite(c0), 'two layers and raw-mass baseline')
    hidden = features.copy()
    with np.errstate(over='raise', invalid='raise'):
        hidden[:, :SPATIAL_DIMENSION] *= GAINS[kind]
        for i, (weight, bias) in enumerate(layers):
            require(weight.dtype == bias.dtype == np.float64 and weight.shape == ((8, DIMENSION) if i == 0 else (1, 8))
                    and bias.shape == (weight.shape[0],) and np.isfinite(weight).all() and np.isfinite(bias).all(),
                    'exact deployed float64 layer shapes')
            hidden = hidden @ weight.T + bias
            if i == 0:
                hidden = np.maximum(hidden, 0)
        result = c0*features[:, :SPATIAL_DIMENSION].sum(axis=1, dtype=np.float64)+hidden[:, 0]
    require(np.isfinite(result).all(), 'finite signed conditioned predictions')
    return result


def initial_pair(raw, scaled, np):
    """Check actual f32 division, unchanged context/output/biases and c0.

The division by53 need not preserve a float32 function bitwise; numerical
initial-function parity is a separately recorded and independently read check.
"""
    checkpoint(raw, 'gain1', float(raw['c0']), np)
    checkpoint(scaled, 'gain53', float(raw['c0']), np)
    expected = raw['weight_0'].copy()
    expected[:, :SPATIAL_DIMENSION] /= np.float32(53)
    require(scaled['weight_0'].tobytes() == expected.tobytes(), 'exact spatial-only inverse f32 compensation')
    for key in ('weight_1', 'bias_0', 'bias_1', 'c0'):
        require(raw[key].tobytes() == scaled[key].tobytes(), 'identical paired output/bias/baseline tensors')
    require(not np.count_nonzero(raw['bias_0']) and not np.count_nonzero(raw['bias_1']), 'zero fresh initial biases')
    return True


class Work(C.Work):
    def finish(self):
        require(next(self.rows, None) is None, 'no extra work calls')
        expected = {'model_initialization': 6, 'optimizer_initialization': 6, 'checkpoint_export': 12,
                    'optimizer_update': 21120, 'parity_restore': 6, 'parity_forward': 12, 'saved_prediction': 162,
                    'initial_pair_initialization': 6, 'initial_pair_export': 6, 'initial_pair_forward': 132}
        require({k: v['returned'] for k, v in self.counts.items()} == expected, 'exact six-fit channel counts')
        return self.counts


count_parameters = C.count_parameters


def pairing_record(record, seed, probe, predictions, initial_sha256, np):
    require(set(record) == {'seed', 'features_sha256', 'rows', 'predictions', 'maximum_difference', 'tolerance', 'passed', 'initial_sha256'}
            and record['seed'] == seed and record['rows'] == len(probe)
            and record['features_sha256'] == hashlib.sha256(probe.tobytes()).hexdigest()
            and record['initial_sha256'] == initial_sha256
            and record['tolerance'] == {'absolute': 1e-6, 'relative': 1e-6}
            and set(record['predictions']) == set(GAINS), 'complete all-TRAIN initial pairing metadata')
    saved = {kind: np.asarray(record['predictions'][kind], np.float64) for kind in GAINS}
    for kind in GAINS:
        expected, observed = predictions[kind], saved[kind]
        require(observed.shape == expected.shape == (len(probe),) and np.isfinite(observed).all()
                and np.all(np.abs(observed-expected) <= 1e-10+1e-10*np.abs(expected)),
                'independent initial prediction replay')
    difference = np.abs(saved['gain1']-saved['gain53'])
    require(type(record['maximum_difference']) in (float, int) and record['maximum_difference'] >= 0,
            'nonnegative initial-pair discrepancy')
    close(record['maximum_difference'], float(difference.max()), 'initial-pair maximum difference')
    passed = bool(np.all(difference <= 1e-6+1e-6*np.abs(saved['gain1'])))
    require(record['passed'] is passed and passed, 'unchanged initial-function tolerance predicate')
    return float(difference.max())

def aggregate(fits, floor):
    require([row['fit_id'] for row in fits] == [f'{kind}@{seed}' for seed in SEEDS for kind in WIDTHS],
            'all six final fits in seed-then-kind order')
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
    for kind in ('gain53',):
        for seed in SEEDS:
            candidate, narrow = metrics[f'{kind}@{seed}'], metrics[f'gain1@{seed}']
            for name, value, threshold in (
                ('train_excess', candidate['train']['mse_normalized']-floor, .8*(narrow['train']['mse_normalized']-floor)),
                ('valid_mse', candidate['valid']['mse_normalized'], .9*narrow['valid']['mse_normalized']),
            ):
                checks.append({'name': f'{kind}.{seed}.{name}', 'value': value, 'threshold': threshold, 'passes': value <= threshold})
    return {'checks': checks, 'conditioning_screen_passed': all(row['passes'] for row in checks)}


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
            and launch['cap_seconds'] == plan['limits']['native_seconds'] == 600
            and launch['deadline_ns'] == launch['started_ns']+600*10**9
            and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
            and terminal['elapsed_ns'] == terminal['finished_ns']-launch['started_ns']
            and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9, 'strict native parent/worker enclosure')


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
        require(plan['version'] == 'otto-conditioning-v1' and plan['mode'] == 'study'
                and plan['status'] == 'frozen_before_execution' and plan['independent_audit_limits'] == LIMITS,
                'fixed full-study and independent-audit allocation')
        own = str(Path(__file__).resolve().relative_to(ROOT))
        require(plan['sources'][own] == digest(Path(__file__), self.check)['sha256']
                and 'tests/test_audit_otto_conditioning.py' in plan['sources'] and plan['sources'][CLOCK] == CLOCK_PIN
                and plan['sources'][HELPER] == HELPER_PIN,
                'prospectively pinned independent source and tests')
        for name, pin in plan['sources'].items():
            require(digest(ROOT/name, self.check)['sha256'] == pin, 'complete unchanged source closure')
        require(digest(args.terminal, self.check)['sha256'] == args.terminal_sha256, 'external successful parent pin')
        worker = manifest(args.run, args.receipt_sha256, payload_names(), self.check)
        require(worker['version'] == 'otto-conditioning-v1' and worker['mode'] == 'study'
                and worker['sources'] == plan['sources'] and worker['inputs'] == plan['inputs']
                and worker['limits'] == plan['limits'] and worker['plan_sha256'] == args.plan_sha256
                and worker['completed_fits'] == 6 and worker['pending'] == [] and worker['parity_passed'] is True
                and worker['initial_pairing_passed'] is True
                and all(worker[k] == 0 for k in ('external_model_calls', 'native_steps', 'native_resets')),
                'complete original study before decoding numerical payloads')
        require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= plan['limits']['rss_bytes']
                and sum(p.stat().st_size for p in args.run.iterdir()) <= plan['limits']['output_bytes'], 'saved worker resource bounds')
        check_terminal(plan, args.plan, worker, read(args.run/'started.json'), read(args.terminal), args.run, self.check)
        # The sole producer reuse authenticates immutable lineage/runtime only.
        producer = load(ROOT/RUNNER, '_conditioning_audit_lineage')
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

    def predict(self, x, layers, c0, kind):
        self.check()
        self.readout_calls += 1
        result = conditioned_predict(x, layers, c0, kind, self.np)
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
        streams = {name: iter(lines(run/name, self.check)) for name in ('initial-pairing.jsonl', 'initializations.jsonl', 'epoch-orders.jsonl',
                   'updates.jsonl', 'fit-curves.jsonl', 'fits.jsonl', 'parity.jsonl')}
        work = Work(lines(run/'work.jsonl', self.check))
        fits, initial_archives = [], {}
        probe = np.concatenate([data['train'][0][:8].astype(np.float64), data['valid'][0][:8].astype(np.float64),
                                np.zeros((1, DIMENSION)), data['train'][0][:1].astype(np.float64)*.5])
        initial_probe = np.concatenate([x.astype(np.float64), np.zeros((1, DIMENSION)), x[:1].astype(np.float64)*.5])
        pairing_seconds, pairing_differences = 0., {}
        for seed in SEEDS:
            predictions = {}
            for kind in WIDTHS:
                context = {'seed': seed, 'kind': kind, 'phase': 'initial_pair'}
                pairing_seconds += work.take('initial_pair_initialization', context)
                pairing_seconds += work.take('initial_pair_export', context)
                initial_archive = self.arrays(run/f'initial-{kind}-{seed}.npz')
                initial_archives[(seed, kind)] = initial_archive
                layers = checkpoint(initial_archive, kind, c0, np)
                values = []
                for offset in range(0, len(initial_probe), 256):
                    pairing_seconds += work.take('initial_pair_forward', {**context, 'offset': offset})
                    values.append(self.predict(initial_probe[offset:offset+256], layers, c0, kind))
                predictions[kind] = np.concatenate(values)
            initial_pair(initial_archives[(seed, 'gain1')], initial_archives[(seed, 'gain53')], np)
            initial_pins = {kind: self.worker['files'][f'initial-{kind}-{seed}.npz']['sha256'] for kind in WIDTHS}
            pairing_differences[str(seed)] = pairing_record(next(streams['initial-pairing.jsonl']), seed, initial_probe, predictions, initial_pins, np)
        del initial_probe
        for seed in SEEDS:
            for kind in WIDTHS:
                self.check()
                fit_id, context = f'{kind}@{seed}', {'fit_id': f'{kind}@{seed}'}
                fit = next(streams['fits.jsonl'])
                require((fit['fit_id'], fit['kind'], fit['seed'], fit['rows'], fit['epochs']) == (fit_id, kind, seed, 5589, 80), 'fixed final fit identity')
                initial_path, final_path = run/f'initial-{kind}-{seed}.npz', run/f'final-{kind}-{seed}.npz'
                initial_archive = initial_archives[(seed, kind)]
                initial = checkpoint(initial_archive, kind, c0, np)
                require(all(not np.count_nonzero(bias) for _, bias in initial), 'all fresh initial biases are zero')
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
                independent = self.predict(probe, layers, c0, kind)
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
                        value = self.predict(vx[offset:offset+256].astype(np.float64), layers, c0, kind)
                        self.equal_array(saved[offset:offset+256], value, 'all independent cached-input predictions')
                    # Independently reduce authenticated saved vectors after numerical replay.
                    stats[split] = scalar_metrics(saved, vy, np)
                close(fit['metrics'], stats, 'final TRAIN/VALID metrics')
                fits.append({**fit, 'metrics': stats})
                del initial, layers, predictions
        for seed in SEEDS:
            initial_pair(initial_archives[(seed, 'gain1')], initial_archives[(seed, 'gain53')], np)
        for stream in streams.values():
            require(next(stream, None) is None, 'no extra fit/order/update/parity rows')
        calls = work.finish()
        close(self.worker['calls'], calls, 'full attempted/returned work and costs')
        require(math.fsum(fit['fit_seconds'] for fit in fits)+pairing_seconds <= self.worker['wall_seconds'],
                'disjoint fit and initial-pair work bounded by actual worker elapsed')
        result = aggregate(fits, floor['irreducible_mse_normalized'])
        summary = read(run/'summary.json')
        require(all(type(summary[key]) in (float, int) and math.isfinite(summary[key]) and summary[key] >= 0
                    for key in ('preparation_seconds', 'initial_pairing_seconds'))
                and summary['initial_pairing_seconds'] >= pairing_seconds
                and math.fsum(fit['fit_seconds'] for fit in fits)+summary['preparation_seconds']
                    +summary['initial_pairing_seconds'] <= self.worker['wall_seconds'], 'disjoint paid preparation, pairing and fit phases')
        expected = {'version': 'otto-conditioning-v1', 'mode': 'study', 'rows': {'train': 5589, 'valid': 1109},
                    'alias_floor': floor, 'metrics': {fit['fit_id']: fit['metrics'] for fit in fits}, **result,
                    'initial_pairing_passed': True, 'preparation_seconds': summary['preparation_seconds'],
                    'initial_pairing_seconds': summary['initial_pairing_seconds'],
                    'qualification_projection_seconds': None, 'training_costs': {fit['fit_id']: fit['fit_seconds'] for fit in fits},
                    'learned_architecture_advantage_established': False,
                    'scope': 'Fixed-data optimization control only; fresh autonomous evidence is separately required regardless of scalar gate.'}
        close(summary, expected, 'all summary numbers and exact six decisions')
        require(self.readout_calls == 300 and self.readout_rows == 73842, 'complete independent initial/final/parity readout allocation')
        return {**expected, 'audit_version': VERSION, 'agreement': True, 'scope': SCOPE,
                'reconstructed_cache': evidence, 'recorded_calls': calls,
                'independent_readout_calls': self.readout_calls, 'independent_readout_rows': self.readout_rows,
                'initial_pairing_maximum_differences': pairing_differences,
                'maximum_prediction_difference': self.maximum_prediction_difference}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists()
                and not any(p.is_symlink() for p in self.out.parents), 'exclusive nonsymlink audit output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(digest(ROOT/CLOCK)['sha256'] == CLOCK_PIN, 'qualified native clock')
            self.clock = load(ROOT/CLOCK, '_conditioning_audit_clock').SuspendClock()
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
