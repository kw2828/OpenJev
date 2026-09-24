"""Independent chronological regression screen; never imports its producer.

The reference equations use precision/information updates and direct Cholesky
solves. Labels enter only at their declared release hour. Missing observations
never collapse the hourly clock or grant access to a future response.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

FEATURES = ('PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'T', 'RH')
KINDS = ('s2linear', 's2quadratic', 'linear7', 'quadratic7')
METHODS = KINDS + ('rls-linear7-lambda1', 'rls-linear7-lambda.995',
                   'rls-quadratic7-lambda1', 'rls-quadratic7-lambda.995', 'persistence')
ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sensor-screen-audit-v1'
CSV_SHA = '13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996'
SOURCES = {'src/openjev/research/sensor_data.py', 'scripts/sensor_screen.py',
           'scripts/audit_sensor_screen.py', 'tests/test_sensor_data.py',
           'tests/test_sensor_screen.py', 'tests/test_audit_sensor_screen.py',
           'research/sensor-screen-protocol.md'}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COLUMNS = ('Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
           'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
           'PT08.S5(O3)', 'T', 'RH', 'AH')
RTOL, ATOL = 1e-7, 1e-8
RIDGE = 1e-6
DELAY = 24


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalization(x, eligible):
    """Population standard deviation of the explicitly admitted FIT inputs."""
    require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 2
            and x.shape[1] == 7, 'seven float64 public features')
    require(isinstance(eligible, np.ndarray) and eligible.dtype == np.bool_
            and eligible.shape == (len(x),), 'explicit FIT normalizer mask')
    rows = x[eligible]
    require(len(rows) > 1 and np.isfinite(rows).all(), 'complete finite FIT normalizer rows')
    mean = rows.mean(axis=0)
    scale = np.sqrt(np.mean((rows - mean)**2, axis=0))
    require(np.all(scale > 0) and np.isfinite(scale).all(), 'nonconstant finite FIT scales')
    return mean, scale


def design(x, mean, scale, kind):
    require(kind in KINDS, 'fixed regression design')
    require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 2
            and x.shape[1] == 7 and np.isfinite(x).all(), 'complete public feature rows')
    require(isinstance(mean, np.ndarray) and isinstance(scale, np.ndarray)
            and mean.dtype == scale.dtype == np.float64 and mean.shape == scale.shape == (7,)
            and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0),
            'frozen finite normalization')
    standardized = (x - mean) / scale
    columns = [np.ones(len(x))]
    if kind.startswith('s2'):
        columns.append(standardized[:, 1])
        if kind == 's2quadratic':
            columns.append(standardized[:, 1]**2)
    else:
        columns.extend(standardized[:, i] for i in range(7))
        if kind == 'quadratic7':
            columns.extend(standardized[:, i] * standardized[:, j]
                           for i in range(7) for j in range(i, 7))
    result = np.column_stack(columns)
    require(np.isfinite(result).all(), 'finite polynomial design')
    return result


def precision_fit(features, target):
    require(isinstance(features, np.ndarray) and features.dtype == np.float64
            and features.ndim == 2 and len(features) > 0 and np.isfinite(features).all(),
            'finite nonempty regression design')
    require(isinstance(target, np.ndarray) and target.dtype == np.float64
            and target.shape == (len(features),) and np.isfinite(target).all(), 'finite admitted FIT labels')
    # Penalize the intercept too, exactly as registered; no unpenalized block.
    precision = RIDGE * np.eye(features.shape[1])
    information = np.zeros(features.shape[1])
    for row, label in zip(features, target, strict=True):
        precision += np.outer(row, row)
        information += row * label
    coefficient = cho_solve(cho_factor(precision, lower=True), information)
    return {'precision': precision, 'information': information, 'coefficient': coefficient}


def clock_inputs(timestamps, x, y):
    require(isinstance(timestamps, np.ndarray) and timestamps.dtype == np.int64
            and timestamps.ndim == 1 and len(timestamps) > 0 and np.all(np.diff(timestamps) > 0),
            'strict unique ascending integer-hour timestamps')
    require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.shape == (len(timestamps), 7)
            and not np.isinf(x).any(), 'seven public features with NaN missingness only')
    require(isinstance(y, np.ndarray) and y.dtype == np.float64 and y.shape == timestamps.shape
            and not np.isinf(y).any(), 'one label with NaN missingness only')


def replay(timestamps, x, y, *, start_hour, end_hour, mean, scale, fit_mask, check=lambda: None):
    """Return every source-row prediction and final states under the fixed delay.

    The FIT eligibility mask is checked against strict release-before-start.
    At each screen hour, including hours with no input row, discount once; then
    reveal the due label, then predict. Future labels are used for scoring only
    outside this routine. Source rows outside SCREEN retain NaN predictions.
    """
    clock_inputs(timestamps, x, y)
    require(type(start_hour) is int and type(end_hour) is int and start_hour < end_hour,
            'positive chronological screen interval')
    complete_x = np.isfinite(x).all(axis=1)
    complete = complete_x & np.isfinite(y)
    require(isinstance(fit_mask, np.ndarray) and fit_mask.dtype == np.bool_
            and fit_mask.shape == timestamps.shape
            and np.array_equal(fit_mask, complete & (timestamps + DELAY < start_hour)),
            'exact complete labels released strictly before screen')
    require(np.any(fit_mask), 'nonempty available FIT labels')
    designs = {kind: design(x[complete_x], mean, scale, kind) for kind in KINDS}
    complete_positions = np.flatnonzero(complete_x)
    lookup = {int(position): j for j, position in enumerate(complete_positions)}
    initial = {kind: precision_fit(designs[kind][[lookup[int(i)] for i in np.flatnonzero(fit_mask)]], y[fit_mask])
               for kind in KINDS}
    states = {}
    for kind in ('linear7', 'quadratic7'):
        for suffix, forgetting in (('1', 1.), ('.995', .995)):
            method = 'rls-' + kind + '-lambda' + suffix
            states[method] = {'precision': initial[kind]['precision'].copy(),
                              'information': initial[kind]['information'].copy(),
                              'forgetting': forgetting, 'kind': kind}
    predictions = {method: np.full(len(timestamps), np.nan) for method in METHODS}
    source_at = {int(hour): i for i, hour in enumerate(timestamps)}
    past = np.flatnonzero(np.isfinite(y) & (timestamps + DELAY < start_hour))
    persistence = float(y[past[-1]])
    persistence_initial = persistence
    assimilated, label_releases = [], []
    prediction_indices = []
    for hour in range(start_hour, end_hour):
        check()
        due = source_at.get(hour - DELAY)
        current = source_at.get(hour)
        revealed = due is not None and complete[due]
        if due is not None and np.isfinite(y[due]):
            persistence = float(y[due])
            label_releases.append(due)
        for state in states.values():
            state['precision'] *= state['forgetting']
            state['information'] *= state['forgetting']
            if revealed:
                row = designs[state['kind']][lookup[due]]
                state['precision'] += np.outer(row, row)
                state['information'] += row * y[due]
        if revealed:
            assimilated.append(due)
        if current is None or not complete_x[current]:
            continue
        prediction_indices.append(current)
        predictions['persistence'][current] = persistence
        for kind in KINDS:
            row = designs[kind][lookup[current]]
            predictions[kind][current] = row @ initial[kind]['coefficient']
        for method, state in states.items():
            row = designs[state['kind']][lookup[current]]
            coefficient = cho_solve(cho_factor(state['precision'], lower=True), state['information'])
            predictions[method][current] = row @ coefficient
    for prediction in predictions.values():
        require(np.isfinite(prediction[prediction_indices]).all(), 'finite chronological predictions')
    return {'predictions': predictions, 'initial': initial,
            'final': {method: {'precision': state['precision'], 'information': state['information']}
                      for method, state in states.items()},
            'assimilated_indices': np.asarray(assimilated, dtype=np.int64),
            'prediction_indices': np.asarray(prediction_indices, dtype=np.int64),
            'label_release_indices': np.asarray(label_releases, dtype=np.int64),
            'persistence_initial': persistence_initial, 'persistence_final': persistence,
            'clock_hours': end_hour - start_hour}


def queue_at(timestamps, x, boundary):
    by_hour = {int(t): i for i, t in enumerate(timestamps)}
    queue = np.full((DELAY, 7), np.nan)
    for hour in range(boundary - DELAY, boundary):
        if hour in by_hour:
            queue[hour % DELAY] = x[by_hour[hour]]
    return queue


def reconstruct(timestamps, x, y, *, start_hour, end_hour, check=lambda: None):
    clock_inputs(timestamps, x, y)
    eligible = np.isfinite(x).all(1) & np.isfinite(y) & (timestamps + DELAY < start_hour)
    mean, scale = normalization(x, eligible)
    reference = replay(timestamps, x, y, start_hour=start_hour, end_hour=end_hour,
                       mean=mean, scale=scale, fit_mask=eligible, check=check)
    target_std = float(np.sqrt(np.mean((y[eligible] - y[eligible].mean())**2)))
    states = {'mean': mean, 'scale': scale, 'fit_mask': eligible,
              'initial_queue': queue_at(timestamps, x, start_hour),
              'fit_target_std': np.asarray(target_std),
              'assimilated_indices': reference['assimilated_indices'],
              'revealed_indices': reference['label_release_indices'],
              'prediction_indices': reference['prediction_indices'],
              'final_persistence': np.asarray(reference['persistence_final'])}
    resources = {'persistence': 17}
    dimensions = {'s2linear': 2, 's2quadratic': 3, 'linear7': 8, 'quadratic7': 36}
    for kind, row in reference['initial'].items():
        for key, name in (('precision', 'A'), ('information', 'b'), ('coefficient', 'coef')):
            states['initial-' + kind + '-' + name] = row[key]
        resources[kind] = 8 * dimensions[kind] + 14 * 8 + 1
    for method, row in reference['final'].items():
        states['final-' + method + '-A'] = row['precision']
        states['final-' + method + '-b'] = row['information']
        states['final-' + method + '-queue'] = queue_at(timestamps, x, end_hour)
        states['final-' + method + '-clock'] = np.asarray(end_hour, dtype=np.int64)
        dimension = len(row['information'])
        resources[method] = 8 * (dimension**2 + dimension + 14 + 24 * 7 + 2) + 1
    return reference['predictions'], states, resources


def score(timestamps, x, y, predictions, states, *, start_hour, june_hour, end_hour):
    """All nine complete-case monthly scores and the fixed six-part screen."""
    require(set(predictions) == set(METHODS), 'all nine methods without selection')
    complete = np.isfinite(x).all(1) & np.isfinite(y)
    scores, counts, best = {}, {}, {}
    for month, lower, upper in (('may', start_hour, june_hour), ('june', june_hour, end_hour)):
        mask = complete & (timestamps >= lower) & (timestamps < upper)
        counts[month] = int(np.count_nonzero(mask))
        scores[month] = {}
        for method in METHODS:
            require(predictions[method].dtype == np.float64 and predictions[method].shape == y.shape
                    and np.isfinite(predictions[method][mask]).all(), 'complete finite scored predictions')
            residuals = predictions[method][mask] - y[mask]
            scores[month][method] = {'rmse': float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
                                     'mae': float(np.mean(np.abs(residuals))) if len(residuals) else None}
        best[month] = min(scores[month][m]['rmse'] for m in KINDS) if counts[month] else None
    fit_rows = int(np.count_nonzero(states['fit_mask']))
    target_std = float(states['fit_target_std'])
    require(np.isfinite(target_std) and target_std >= 0, 'finite FIT target population deviation')
    conditions = [
        {'name': 'fit_rows_at_least_512', 'passed': fit_rows >= 512},
        {'name': 'may_rows_at_least_256', 'passed': counts['may'] >= 256},
        {'name': 'june_rows_at_least_256', 'passed': counts['june'] >= 256},
    ]
    for month in ('may', 'june'):
        conditions.append({'name': month + '_static_rmse_above_1pct_fit_std',
                           'passed': best[month] is not None and best[month] > .01 * target_std})
    adaptive = [m for m in METHODS if m.startswith('rls-')]
    qualifiers = [method for method in adaptive if all(best[month] is not None and
                  scores[month][method]['rmse'] <= .9 * best[month] for month in ('may', 'june'))]
    conditions.append({'name': 'same_rls_beats_best_static_by_10pct_both_months', 'passed': bool(qualifiers)})
    passed = sum(row['passed'] for row in conditions)
    return {'version': 'sensor-screen-v1', 'scope': 'TRAIN-only benchmark qualification, not held-out results',
            'fit_rows': fit_rows, 'fit_target_std': target_std, 'month_rows': counts, 'metrics': scores,
            'best_static_rmse': best, 'adaptive_qualifiers': qualifiers, 'conditions': conditions,
            'passed': passed, 'total': 6, 'outcome': 'QUALIFIES_LEARNED_MEMORY_SCREEN' if passed == 6
            else 'REJECT_BENZENE_MEMORY_BENCHMARK'}


def hour(text):
    # Naive source calendar, intentionally no timezone/DST inference.
    epoch = datetime(1970, 1, 1)  # noqa: DTZ001
    return int((datetime.fromisoformat(text) - epoch).total_seconds() // 3600)


START, JUNE, END = (hour(value) for value in ('2004-05-01', '2004-06-01', '2004-07-01'))


def config():
    return {'version': 'sensor-screen-v1', 'start_hour': START, 'june_hour': JUNE, 'end_hour': END,
            'delay_hours': DELAY, 'ridge': RIDGE, 'features': list(FEATURES), 'target': 'C6H6(GT)',
            'methods': list(METHODS), 'min_fit_rows': 512, 'min_month_rows': 256,
            'min_static_error_std_fraction': .01, 'max_adaptive_static_rmse_ratio': .9,
            'wall_cap_seconds': 300, 'held_out_numeric_access': False}


def parse_train(raw):
    """Independent raw CSV reconciliation; held-out measurement cells stay strings."""
    require(type(raw) is bytes, 'raw CSV bytes')
    try:
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), delimiter=';', strict=True)
        header = tuple(value.strip() for value in next(reader))
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError('valid nonempty semicolon CSV') from error
    require(header[:15] == COLUMNS and all(not v for v in header[15:]), 'original CSV columns')
    rows, previous, trailing = [], None, False
    numeric_indices = [COLUMNS.index(name) for name in (*FEATURES, 'C6H6(GT)')]
    for row_id, cells in enumerate(reader, start=1):
        cells = [value.strip() for value in cells]
        if not any(cells):
            trailing = True
            continue
        require(not trailing and len(cells) == len(header) and all(not v for v in cells[15:]),
                'source row geometry and terminal blank rows')
        require(re.fullmatch(r'\d{2}/\d{2}/\d{4}', cells[0]) is not None
                and re.fullmatch(r'\d{2}\.\d{2}\.\d{2}', cells[1]) is not None, 'source date/time syntax')
        stamp = datetime.strptime(cells[0] + ' ' + cells[1], '%d/%m/%Y %H.%M.%S')  # noqa: DTZ007
        require(stamp.minute == stamp.second == 0, 'integer source calendar hour')
        current = hour(stamp.isoformat())
        require(previous is None or current > previous, 'strict original chronology')
        previous = current
        if not hour('2004-03-01') <= current < END:
            continue  # No numerical conversion or fit on July or later labels.
        values = []
        for index in numeric_indices:
            value = cells[index]
            if not value or re.fullmatch(r'-200(?:,0+)?', value):
                values.append(np.nan)
            else:
                require(re.fullmatch(r'[+-]?\d+(?:,\d+)?', value) is not None, 'finite decimal-comma TRAIN value')
                parsed = float(value.replace(',', '.'))
                require(math.isfinite(parsed), 'finite parsed TRAIN value')
                values.append(parsed)
        rows.append((current, row_id, values))
    require(bool(rows), 'nonempty TRAIN interval')
    timestamps = np.array([row[0] for row in rows], dtype=np.int64)
    row_ids = np.array([row[1] for row in rows], dtype=np.int64)
    x = np.array([row[2][:-1] for row in rows], dtype=np.float64)
    y = np.array([row[2][-1] for row in rows], dtype=np.float64)
    return {'timestamp_hours': timestamps, 'row_ids': row_ids, 'x': x, 'y': y,
            'valid_inputs': np.isfinite(x), 'valid_target': np.isfinite(y)}


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary original evidence file')
    contents = path.read_bytes()
    return {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, message):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message + ' schema')
        for key in expected:
            compare(actual[key], expected[key], message + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), message + ' list')
        for left, right in zip(actual, expected, strict=True):
            compare(left, right, message)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), message + ' numerical agreement')
    else:
        require(type(actual) is type(expected) and actual == expected, message + ' exact value')


def array_agreement(actual, expected, message, *, exact=False):
    require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
            and actual.shape == expected.shape, message + ' dtype/shape')
    if exact or expected.dtype.kind not in 'fc':
        require(actual.tobytes() == expected.tobytes(), message + ' exact bytes')
    else:
        require(np.array_equal(np.isnan(actual), np.isnan(expected))
                and not np.isinf(actual).any() and not np.isinf(expected).any()
                and np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True), message + ' numerical agreement')


def absolute(value):
    require(type(value) is str, 'string command arguments')
    path = Path(value)
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def closed_receipt(path, source_pins, expected_argv=None):
    path = Path(path)
    receipt = read(path)
    require(set(receipt) == {'state', 'returncode', 'argv', 'elapsed_seconds', 'thread_env',
            'sources_before', 'sources_after', 'log_path', 'log', 'timeout_seconds', 'wrapper'},
            'complete original process receipt')
    require(receipt['state'] == 'EXITED' and type(receipt['returncode']) is int
            and receipt['returncode'] == 0, 'clean original process closure')
    seconds = receipt['elapsed_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 300
            and type(receipt['timeout_seconds']) is int and receipt['timeout_seconds'] == 300,
            'original fixed process cap')
    require(receipt['sources_before'] == receipt['sources_after'] == source_pins,
            'original process source pins before and after')
    require(receipt['thread_env'] == {name: '1' for name in THREADS}, 'original single-thread process')
    require(type(receipt['argv']) is list and all(type(v) is str for v in receipt['argv']), 'original string argv')
    if expected_argv is not None:
        require(receipt['argv'] == expected_argv, 'exact qualification argv')
    log_name = receipt['log_path']
    require(type(log_name) is str and bool(log_name) and Path(log_name).name == log_name,
            'one original sibling log')
    require(descriptor(path.parent / log_name) == receipt['log'], 'original process log pin')
    require(descriptor(ROOT / 'output/sensor-screen-engineering-v1/invoke.py') == receipt['wrapper'],
            'original exclusive process wrapper')
    return receipt


def authenticate(study, csv_path, receipt_path):
    folder = Path(study).resolve()
    csv_path = Path(csv_path).resolve()
    plan = read(folder / 'registration.json')
    require(set(plan) == {'config', 'csv', 'qualification', 'sources', 'environment'}, 'registration schema')
    compare(plan['config'], config(), 'exact registered screen')
    require(set(plan['sources']) == SOURCES, 'complete seven-source closure')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT / name) == expected == descriptor(folder / 'sources' / name),
                'current/snapshot source pins')
    require(plan['csv']['sha256'] == CSV_SHA and descriptor(csv_path) == plan['csv'], 'official original CSV pin')
    compare(plan['environment'], {'python': sys.version, 'numpy': np.__version__,
            'threads': {name: '1' for name in THREADS}}, 'frozen runtime')
    require(all(os.environ.get(name) == '1' for name in THREADS), 'single-thread audit environment')
    qualification = read(folder / 'qualification.json')
    require(descriptor(folder / 'qualification.json') == plan['qualification']
            and qualification['state'] == 'EXITED' and type(qualification['returncode']) is int
            and qualification['returncode'] == 0, 'original qualification closure')
    expected_files = {'registration.json', 'qualification.json', 'train.npz', 'predictions.npz',
                      'states.npz', 'results.json', 'resources.json', 'run.json'} | {'sources/' + s for s in SOURCES}
    manifest = read(folder / 'manifest.json')
    require(set(manifest) == {'files'} and set(manifest['files']) == expected_files, 'complete original file roster')
    require(not any(p.is_symlink() for p in folder.rglob('*')), 'no symlink evidence')
    actual_files = {p.relative_to(folder).as_posix(): descriptor(p) for p in folder.rglob('*')
                    if p.is_file() and p.name != 'manifest.json'}
    require(actual_files == manifest['files'], 'opaque file inventory before numerical access')
    run = read(folder / 'run.json')
    require(set(run) == {'version', 'registration', 'registration_commit', 'csv', 'elapsed_seconds',
            'numeric_segments_loaded', 'models_fitted', 'adaptive_streams', 'clock_hours',
            'source_pins_verified_before_and_after'}, 'complete run receipt')
    fixed = {'version': 'sensor-screen-v1', 'registration': descriptor(folder / 'registration.json'),
             'csv': plan['csv'], 'numeric_segments_loaded': ['train'], 'models_fitted': 4,
             'adaptive_streams': 4, 'clock_hours': END - START, 'source_pins_verified_before_and_after': True}
    compare({key: run[key] for key in fixed}, fixed, 'original run scope')
    seconds = run['elapsed_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 300, 'run elapsed cap')
    terminal = closed_receipt(receipt_path, plan['sources'])
    elapsed = terminal['elapsed_seconds']
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and seconds <= elapsed <= 300,
            'original process elapsed cap')
    command = terminal['argv']
    require(type(command) is list and len(command) == 11 and all(type(v) is str for v in command), 'exact run argv roster')
    require(absolute(command[0]) == ROOT / '.venv/bin/python' and absolute(command[1]) == ROOT / 'scripts/sensor_screen.py'
            and command[2:4] == ['run', '--csv'] and absolute(command[4]) == csv_path
            and command[5] == '--registration' and command[7] == '--qualification'
            and command[9] == '--output' and absolute(command[10]) == folder, 'original run argv and paths')
    registration_path = absolute(command[6])
    require(descriptor(registration_path) == descriptor(folder / 'registration.json')
            and descriptor(absolute(command[8])) == plan['qualification'], 'original admission file joins')
    qualification_argv = ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_sensor_data.py',
                          'tests/test_sensor_screen.py', 'tests/test_audit_sensor_screen.py']
    original_qualification = closed_receipt(absolute(command[8]), plan['sources'], qualification_argv)
    require(original_qualification == qualification, 'copied original qualification identity')
    commit = run['registration_commit']
    require(type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'original registration commit')
    for name, pin in {registration_path.relative_to(ROOT).as_posix(): run['registration'], **plan['sources']}.items():
        contents = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require({'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()} == pin,
                'committed-before-fit source and registration')
    return {'registration': descriptor(folder / 'registration.json'), 'manifest': descriptor(folder / 'manifest.json'),
            'run_receipt': descriptor(receipt_path), 'qualification': plan['qualification'],
            'csv': plan['csv'], 'sources': plan['sources'], 'registration_commit': commit,
            'original_process_seconds': elapsed, 'original_inner_seconds': seconds, 'manifest_files': len(actual_files)}


def load_arrays(path, fields):
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == set(fields), 'complete saved array roster: ' + path.name)
        return {name: values[name].copy() for name in fields}


def audit(study, *, csv_path, run_receipt, check=lambda: None):
    started = time.perf_counter()
    folder = Path(study).resolve()
    check()
    admission = authenticate(folder, csv_path, run_receipt)
    original = parse_train(Path(csv_path).read_bytes())
    saved_train = load_arrays(folder / 'train.npz', original)
    for name, expected in original.items():
        array_agreement(saved_train[name], expected, 'raw TRAIN reconciliation/' + name, exact=True)
    predictions, states, resources = reconstruct(original['timestamp_hours'], original['x'], original['y'],
                                                start_hour=START, end_hour=END, check=check)
    saved_predictions = load_arrays(folder / 'predictions.npz', predictions)
    saved_states = load_arrays(folder / 'states.npz', states)
    for name, expected in predictions.items():
        array_agreement(saved_predictions[name], expected, 'independent delayed prediction/' + name)
    for name, expected in states.items():
        array_agreement(saved_states[name], expected, 'independent state/' + name,
                        exact='queue' in name or name in ('final_persistence',))
    # Score saved predictions after independent agreement, preserving the exact
    # registered threshold comparisons instead of rounding near a decision edge.
    result = score(original['timestamp_hours'], original['x'], original['y'], saved_predictions, states,
                   start_hour=START, june_hour=JUNE, end_hour=END)
    compare(read(folder / 'results.json'), result, 'independent monthly metrics and fixed screen')
    resource = {'logical_persistent_bytes': resources, 'fixed_byte_cap': None,
                'queue_raw_inputs_bytes_per_rls': DELAY * 7 * 8,
                'includes': 'dense precision, information, normalization, queue, decay, clock, type tag',
                'excludes': 'Python/runtime/native workspace, fit-only and audit artifacts, environment',
                'scope': 'No memory-matched architecture or latency claim'}
    compare(read(folder / 'resources.json'), resource, 'complete persistent byte accounting')
    check()
    require(authenticate(folder, csv_path, run_receipt) == admission, 'all original evidence unchanged after audit')
    return {'version': VERSION, 'agreement': True, 'admission': admission, 'results': result,
            'resources': resource, 'counts': {'npz_decodes': 3, 'array_loads': len(original) + len(predictions) + len(states),
            'numerical_segments': ['train'], 'held_out_numeric_values': 0, 'model_calls': 0, 'producer_calls': 0,
            'independent_static_fits': 4, 'independent_adaptive_streams': 4, 'clock_hours': END - START},
            'tolerance': {'rtol': RTOL, 'atol': ATOL}, 'seconds': time.perf_counter() - started,
            'limitations': ['TRAIN-only usefulness screen, not held-out performance or a learned-memory result.',
                            'Hourly label delay is imposed; source timestamps remain naive local calendar hours.',
                            'Pending inputs and dense regression states are counted; no byte-matched architecture claim.',
                            'Metadata and source-time execution are original receipt evidence.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-receipt', required=True, type=Path)
    parser.add_argument('--csv', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not (args.output.parent / 'manifest.json').exists()
            and not args.output.resolve().is_relative_to(args.study.resolve()), 'new audit output outside original study')
    result = audit(args.study, csv_path=args.csv, run_receipt=args.run_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    with (args.output.parent / 'manifest.json').open('x') as stream:
        json.dump({'files': {args.output.name: descriptor(args.output)}}, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'agreement': result['agreement'], 'outcome': result['results']['outcome']}), flush=True)


if __name__ == '__main__':
    main()
