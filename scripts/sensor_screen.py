"""Registered TRAIN-only qualification of delayed sensor-memory research."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from openjev.research.sensor_data import INPUT_COLUMNS, load_train

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sensor-screen-v1'
CSV_SHA = '13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996'
SOURCES = (
    'src/openjev/research/sensor_data.py', 'scripts/sensor_screen.py',
    'scripts/audit_sensor_screen.py', 'tests/test_sensor_data.py',
    'tests/test_sensor_screen.py', 'tests/test_audit_sensor_screen.py',
    'research/sensor-screen-protocol.md',
)
KINDS = ('s2linear', 's2quadratic', 'linear7', 'quadratic7')
RLS = ('rls-linear7-lambda1', 'rls-linear7-lambda.995',
       'rls-quadratic7-lambda1', 'rls-quadratic7-lambda.995')
METHODS = KINDS + RLS + ('persistence',)
DELAY = 24
RIDGE = 1e-6
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hour(text):
    return int(np.datetime64(text, 'h').astype(np.int64))


START, JUNE, END = map(hour, ('2004-05-01', '2004-06-01', '2004-07-01'))


def pin(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_json(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def config():
    return {'version': VERSION, 'start_hour': START, 'june_hour': JUNE, 'end_hour': END,
            'delay_hours': DELAY, 'ridge': RIDGE, 'features': list(INPUT_COLUMNS),
            'target': 'C6H6(GT)', 'methods': list(METHODS), 'min_fit_rows': 512,
            'min_month_rows': 256, 'min_static_error_std_fraction': .01,
            'max_adaptive_static_rmse_ratio': .9, 'wall_cap_seconds': 300,
            'held_out_numeric_access': False}


def feature_map(x, mean, scale, kind):
    require(kind in KINDS, 'unknown design')
    z = (np.asarray(x, dtype=np.float64) - mean) / scale
    require(z.ndim == 2 and z.shape[1] == 7 and np.isfinite(z).all(), 'finite seven-input design')
    one = np.ones((len(z), 1))
    if kind == 's2linear':
        return np.concatenate((one, z[:, 1:2]), axis=1)
    if kind == 's2quadratic':
        return np.concatenate((one, z[:, 1:2], z[:, 1:2]**2), axis=1)
    if kind == 'linear7':
        return np.concatenate((one, z), axis=1)
    products = np.column_stack([z[:, i]*z[:, j] for i in range(7) for j in range(i, 7)])
    return np.concatenate((one, z, products), axis=1)


class DelayedFeed:
    """Environment boundary: each step exposes x_now and only y_now-delay.

    Actors never receive the full target array. Missing calendar hours remain
    steps. Initial labels are strictly released before start, leaving the
    boundary label for the first step rather than assimilating it twice.
    """
    def __init__(self, timestamps, x, y, start):
        require(timestamps.dtype == np.int64 and timestamps.ndim == 1 and
                np.all(np.diff(timestamps) > 0), 'unique chronological integer hours')
        require(x.dtype == np.float64 and x.shape == (len(timestamps), 7) and
                y.dtype == np.float64 and y.shape == timestamps.shape and
                not np.isinf(x).any() and not np.isinf(y).any(), 'finite or NaN measurements')
        self._timestamps, self._x, self._y = timestamps, x, y
        self._index = {int(t): i for i, t in enumerate(timestamps)}
        self._next = start
        self._start = start
        self._initialized = False

    def initialization(self):
        require(self._next == self._start and not self._initialized, 'initialize only once before replay')
        self._initialized = True
        eligible = (self._timestamps + DELAY < self._next)
        fit = eligible & np.isfinite(self._x).all(axis=1) & np.isfinite(self._y)
        labeled = np.flatnonzero(eligible & np.isfinite(self._y))
        require(fit.sum() >= 2 and len(labeled) > 0, 'usable released FIT rows required')
        queue = np.full((DELAY, 7), np.nan)
        for t in range(self._next - DELAY, self._next):
            row = self._index.get(t)
            if row is not None:
                queue[t % DELAY] = self._x[row]
        return (fit, self._x[fit].copy(), self._y[fit].copy(), queue,
                float(self._y[labeled[-1]]))

    def step(self, current_hour):
        require(current_hour == self._next, 'feed cannot skip, replay, or reveal a future hour')
        now, due = self._index.get(current_hour), self._index.get(current_hour - DELAY)
        x = np.full(7, np.nan) if now is None else self._x[now].copy()
        y = np.nan if due is None else float(self._y[due])
        self._next += 1
        return x, y, now, due


class Adaptive:
    def __init__(self, kind, forgetting, precision, information, mean, scale, queue, start):
        self.kind, self.forgetting = kind, np.float64(forgetting)
        self.a, self.b = precision.copy(), information.copy()
        self.mean, self.scale, self.queue = mean.copy(), scale.copy(), queue.copy()
        self.next_hour = np.int64(start)

    def step(self, current_hour, x, due_y):
        require(current_hour == self.next_hour, 'adaptive clock mismatch')
        self.a *= self.forgetting
        self.b *= self.forgetting
        delayed_x = self.queue[current_hour % DELAY]
        assimilated = np.isfinite(delayed_x).all() and np.isfinite(due_y)
        if assimilated:
            phi = feature_map(delayed_x[None], self.mean, self.scale, self.kind)[0]
            self.a += np.outer(phi, phi)
            self.b += phi * due_y
        prediction = np.nan
        if np.isfinite(x).all():
            phi = feature_map(x[None], self.mean, self.scale, self.kind)[0]
            prediction = float(phi @ np.linalg.solve(self.a, self.b))
        self.queue[current_hour % DELAY] = x
        self.next_hour += 1
        return prediction, bool(assimilated)

    def logical_bytes(self):
        # Actual dense matrices, duplicated normalization, raw pending inputs,
        # decay scalar, clock and one-byte design tag; no packed-matrix claim.
        return sum(a.nbytes for a in (self.a, self.b, self.mean, self.scale, self.queue,
                                     self.forgetting, self.next_hour)) + 1


def run_arrays(timestamps, x, y, *, start=START, end=END, check=lambda: None):
    """Causal producer; arrays must already be admitted TRAIN data."""
    feed = DelayedFeed(timestamps, x, y, start)
    fit, fit_x, fit_y, pending, last_y = feed.initialization()
    mean, scale = fit_x.mean(axis=0), fit_x.std(axis=0, ddof=0)
    require(np.isfinite(scale).all() and np.all(scale > 0), 'positive FIT feature scales')
    states = {'mean': mean, 'scale': scale, 'fit_mask': fit,
              'initial_queue': pending.copy(), 'fit_target_std': np.asarray(fit_y.std(ddof=0))}
    static, adaptive, resources = {}, {}, {}
    for kind in KINDS:
        phi = feature_map(fit_x, mean, scale, kind)
        a = phi.T @ phi + RIDGE * np.eye(phi.shape[1])
        b = phi.T @ fit_y
        coef = np.linalg.solve(a, b)
        states[f'initial-{kind}-A'], states[f'initial-{kind}-b'] = a.copy(), b.copy()
        states[f'initial-{kind}-coef'] = coef.copy()
        static[kind] = coef
        resources[kind] = int(coef.nbytes + mean.nbytes + scale.nbytes + 1)
        if kind in ('linear7', 'quadratic7'):
            for name, decay in (('1', 1.), ('.995', .995)):
                method = f'rls-{kind}-lambda{name}'
                adaptive[method] = Adaptive(kind, decay, a, b, mean, scale, pending, start)
                resources[method] = adaptive[method].logical_bytes()
    resources['persistence'] = 17  # Target scalar, clock, one-byte tag. No input queue.
    predictions = {m: np.full(len(timestamps), np.nan) for m in METHODS}
    assimilated, revealed, prediction_indices = [], [], []
    for t in range(start, end):
        check()
        current_x, due_y, now, due = feed.step(t)
        if np.isfinite(due_y):
            last_y = due_y
            revealed.append(due)
        flags = []
        for name, model in adaptive.items():
            pred, flag = model.step(t, current_x, due_y)
            flags.append(flag)
            if now is not None:
                predictions[name][now] = pred
        require(len(set(flags)) == 1, 'adaptive histories must be matched')
        if flags[0]:
            require(due is not None, 'assimilation requires due source row')
            assimilated.append(due)
        if now is None or not np.isfinite(current_x).all():
            continue
        prediction_indices.append(now)
        for name, coef in static.items():
            predictions[name][now] = feature_map(current_x[None], mean, scale, name)[0] @ coef
        predictions['persistence'][now] = last_y
    states['assimilated_indices'] = np.asarray(assimilated, dtype=np.int64)
    states['revealed_indices'] = np.asarray(revealed, dtype=np.int64)
    states['prediction_indices'] = np.asarray(prediction_indices, dtype=np.int64)
    states['final_persistence'] = np.asarray(last_y)
    for name, model in adaptive.items():
        states[f'final-{name}-A'], states[f'final-{name}-b'] = model.a, model.b
        states[f'final-{name}-queue'] = model.queue
        states[f'final-{name}-clock'] = np.asarray(model.next_hour)
    for values in predictions.values():
        require(np.isfinite(values[prediction_indices]).all(), 'nonfinite valid-input prediction')
    return predictions, states, resources


def summarize(timestamps, x, y, predictions, states):
    require(set(predictions) == set(METHODS), 'all nine methods required')
    complete = np.isfinite(x).all(axis=1) & np.isfinite(y)
    metrics, counts, best = {}, {}, {}
    for month, lower, upper in (('may', START, JUNE), ('june', JUNE, END)):
        eligible = complete & (timestamps >= lower) & (timestamps < upper)
        counts[month] = int(eligible.sum())
        metrics[month] = {}
        for method, pred in predictions.items():
            require(np.isfinite(pred[eligible]).all(), 'all complete rows must be scored')
            error = pred[eligible] - y[eligible]
            metrics[month][method] = {
                'rmse': float(np.sqrt(np.mean(error**2))) if len(error) else None,
                'mae': float(np.mean(np.abs(error))) if len(error) else None,
            }
        best[month] = min((metrics[month][m]['rmse'] for m in KINDS),
                          default=None) if counts[month] else None
    target_std = float(states['fit_target_std'])
    conditions = [
        {'name': 'fit_rows_at_least_512', 'passed': bool(states['fit_mask'].sum() >= 512)},
        {'name': 'may_rows_at_least_256', 'passed': counts['may'] >= 256},
        {'name': 'june_rows_at_least_256', 'passed': counts['june'] >= 256},
        *[{'name': f'{month}_static_rmse_above_1pct_fit_std',
           'passed': best[month] is not None and best[month] > .01 * target_std}
          for month in ('may', 'june')],
    ]
    winners = [method for method in RLS if all(best[month] is not None and
               metrics[month][method]['rmse'] <= .9 * best[month] for month in ('may', 'june'))]
    conditions.append({'name': 'same_rls_beats_best_static_by_10pct_both_months', 'passed': bool(winners)})
    return {'version': VERSION, 'scope': 'TRAIN-only benchmark qualification, not held-out results',
            'fit_rows': int(states['fit_mask'].sum()), 'fit_target_std': target_std,
            'month_rows': counts, 'metrics': metrics, 'best_static_rmse': best,
            'adaptive_qualifiers': winners, 'conditions': conditions,
            'passed': sum(c['passed'] for c in conditions), 'total': len(conditions),
            'outcome': 'QUALIFIES_LEARNED_MEMORY_SCREEN' if all(c['passed'] for c in conditions)
                       else 'REJECT_BENZENE_MEMORY_BENCHMARK'}


QUALIFICATION_ARGV = ['.venv/bin/python', '-m', 'pytest', '-q',
                      'tests/test_sensor_data.py', 'tests/test_sensor_screen.py',
                      'tests/test_audit_sensor_screen.py']


def authenticate_qualification(path):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    require(receipt['state'] == 'EXITED' and receipt['returncode'] == 0,
            'closed passing qualification required')
    require(receipt['argv'] == QUALIFICATION_ARGV, 'exact qualification test command required')
    current = {s: pin((ROOT/s).read_bytes()) for s in SOURCES}
    require(receipt['sources_before'] == receipt['sources_after'] == current,
            'qualification must cover unchanged registered sources')
    require(receipt['thread_env'] == {k: '1' for k in THREADS}, 'qualification thread environment')
    log_path = Path(path).parent / receipt['log_path']
    require(log_path.parent.resolve() == Path(path).parent.resolve() and
            pin(log_path.read_bytes()) == receipt['log'], 'original qualification log changed')
    return raw


def register(args):
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread environment required')
    raw = Path(args.csv).read_bytes()
    require(pin(raw)['sha256'] == CSV_SHA, 'official CSV pin mismatch')
    qualification = authenticate_qualification(args.qualification)
    write_json(args.registration, {'config': config(), 'csv': pin(raw),
               'qualification': pin(qualification), 'sources': {s: pin((ROOT/s).read_bytes()) for s in SOURCES},
               'environment': {'python': sys.version, 'numpy': np.__version__,
                               'threads': {k: os.environ[k] for k in THREADS}}})


def run(args):
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread environment required')
    registration_path = Path(args.registration).resolve()
    reg_raw = registration_path.read_bytes()
    reg = json.loads(reg_raw)
    require(reg['config'] == config() and set(reg['sources']) == set(SOURCES), 'exact registered configuration')
    qualification_raw = authenticate_qualification(args.qualification)
    require(pin(qualification_raw) == reg['qualification'], 'qualification receipt changed')
    require(all(pin((ROOT/s).read_bytes()) == expected for s, expected in reg['sources'].items()),
            'registered source changed')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    relative = registration_path.relative_to(ROOT).as_posix()
    require(subprocess.check_output(['git', 'show', f'{head}:{relative}'], cwd=ROOT) == reg_raw,
            'registration must be committed before fitting')
    for source, expected in reg['sources'].items():
        require(pin(subprocess.check_output(['git', 'show', f'{head}:{source}'], cwd=ROOT)) == expected,
                'source must match registration commit')
    raw = Path(args.csv).read_bytes()
    require(pin(raw) == reg['csv'] and reg['csv']['sha256'] == CSV_SHA, 'original CSV changed')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def check():
        if time.monotonic() - start > reg['config']['wall_cap_seconds']:
            raise TimeoutError('registered sensor-screen wall cap')
    (output/'sources').mkdir()
    for source in SOURCES:
        dest = output/'sources'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/source).read_bytes())
    (output/'registration.json').write_bytes(reg_raw)
    (output/'qualification.json').write_bytes(qualification_raw)
    rows = load_train(raw)
    require(rows.segment == 'train' and np.all(rows.timestamp_hours < END), 'TRAIN boundary')
    np.savez_compressed(output/'train.npz', timestamp_hours=rows.timestamp_hours,
                        row_ids=rows.row_ids, x=rows.x, y=rows.y,
                        valid_inputs=rows.valid_inputs, valid_target=rows.valid_target)
    predictions, states, resources = run_arrays(rows.timestamp_hours, rows.x, rows.y, check=check)
    np.savez_compressed(output/'predictions.npz', **predictions)
    np.savez_compressed(output/'states.npz', **states)
    result = summarize(rows.timestamp_hours, rows.x, rows.y, predictions, states)
    write_json(output/'results.json', result)
    write_json(output/'resources.json', {'logical_persistent_bytes': resources,
               'fixed_byte_cap': None, 'queue_raw_inputs_bytes_per_rls': DELAY*7*8,
               'includes': 'dense precision, information, normalization, queue, decay, clock, type tag',
               'excludes': 'Python/runtime/native workspace, fit-only and audit artifacts, environment',
               'scope': 'No memory-matched architecture or latency claim'})
    check()
    require(all(pin((ROOT/s).read_bytes()) == expected for s, expected in reg['sources'].items()),
            'registered source changed during execution')
    write_json(output/'run.json', {'version': VERSION, 'registration': pin(reg_raw),
               'registration_commit': head, 'csv': pin(raw), 'elapsed_seconds': time.monotonic()-start,
               'numeric_segments_loaded': ['train'], 'models_fitted': 4, 'adaptive_streams': 4,
               'clock_hours': END-START, 'source_pins_verified_before_and_after': True})
    entries = {p.relative_to(output).as_posix(): pin(p.read_bytes())
               for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'manifest.json', {'files': entries})
    check()
    print(json.dumps({'outcome': result['outcome'], 'passed': result['passed'], 'total': result['total']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    parser.add_argument('--csv', required=True)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'register':
        register(args)
    else:
        require(bool(args.output), '--output is required')
        run(args)


if __name__ == '__main__':
    main()
