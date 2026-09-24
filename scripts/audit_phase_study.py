"""Independent saved-output audit for the Silverbox phase-mechanism study.

Inference is implemented here in NumPy, without importing the producer or its
models. Recorded optimization is checked structurally, not replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'phase-study-audit-v1'
PHASE = ('energy_phase', 'fixed_phase', 'nonlinear_readout')
TRAINED = (*PHASE, 'gru16', 'cubic_ar2')
CLASSICAL = ('static_cubic', 'fir128', 'fir512')
METHODS = (*TRAINED, *CLASSICAL)
SEEDS = (7301, 7302, 7303)
PARTITIONS = ('dev_a', 'dev_b')
RTOL = ATOL = 1e-4
RIDGE = 1e-6
BASE_SHAPES = {'rho_logits': (2,), 'angle_logits': (2,), 'input_weights': (2, 2),
               'readout_weights': (2, 2), 'feedthrough': (), 'bias': ()}
GRU_SHAPES = {'gru.weight_ih_l0': (48, 1), 'gru.weight_hh_l0': (48, 16),
              'gru.bias_ih_l0': (48,), 'gru.bias_hh_l0': (48,),
              'readout.weight': (1, 16), 'readout.bias': (1,)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def shapes(arm):
    if arm in PHASE:
        result = dict(BASE_SHAPES)
        if arm == 'energy_phase':
            result.update(phase_logits=(2,), energy_scales_raw=(2,))
        elif arm == 'nonlinear_readout':
            result['cubic_readout_weights'] = (2, 2)
        return result
    if arm == 'gru16':
        return dict(GRU_SHAPES)
    require(arm == 'cubic_ar2', 'known trained arm')
    return {'coefficient': (7,)}


def parameters(values, arm):
    expected = shapes(arm)
    require(type(values) is dict and set(values) == set(expected), 'complete parameter roster')
    for name, shape in expected.items():
        array = values[name]
        require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.shape == shape
                and np.isfinite(array).all(), 'finite float32 parameter: ' + name)
    return {name: value.copy() for name, value in values.items()}


def inputs(values):
    require(isinstance(values, np.ndarray) and values.dtype == np.float32 and values.ndim == 1
            and len(values) > 0 and np.isfinite(values).all(), 'finite float32 input vector')
    return values


def phase_predict(values, u, arm):
    p, u = parameters(values, arm), inputs(u)
    require(arm in PHASE, 'phase arm')
    radius = np.float32(.9999) * expit(p['rho_logits'])
    angle = np.float32(np.pi) * expit(p['angle_logits'])
    if arm == 'energy_phase':
        strength = np.float32(.5) * np.tanh(p['phase_logits'])
        scale = np.logaddexp(np.float32(0), p['energy_scales_raw'])
    state = np.zeros((2, 2), dtype=np.float32)
    result = np.empty(len(u), dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        for index, current in enumerate(u):
            theta = angle
            if arm == 'energy_phase':
                theta = angle + strength * np.tanh(scale * np.sum(state * state, axis=1))
            cosine, sine = np.cos(theta), np.sin(theta)
            rotated = np.stack((cosine * state[:, 0] - sine * state[:, 1],
                                sine * state[:, 0] + cosine * state[:, 1]), axis=1)
            state = radius[:, None] * rotated + current * p['input_weights']
            value = np.sum(state * p['readout_weights'])
            if arm == 'nonlinear_readout':
                cubic = state * np.sum(state * state, axis=1, keepdims=True)
                value = value + np.sum(cubic * p['cubic_readout_weights'])
            result[index] = value + p['feedthrough'] * current + p['bias']
    return result


def gru_predict(values, u):
    p, u = parameters(values, 'gru16'), inputs(u)
    state = np.zeros(16, dtype=np.float32)
    result = np.empty(len(u), dtype=np.float32)
    for index, current in enumerate(u):
        incoming = p['gru.weight_ih_l0'][:, 0] * current + p['gru.bias_ih_l0']
        recurrent = p['gru.weight_hh_l0'] @ state + p['gru.bias_hh_l0']
        ir, iz, inn = np.split(incoming, 3)
        hr, hz, hn = np.split(recurrent, 3)
        reset, update = expit(ir + hr), expit(iz + hz)
        proposal = np.tanh(inn + reset * hn)
        state = (np.float32(1) - update) * proposal + update * state
        result[index] = (p['readout.weight'] @ state + p['readout.bias'])[0]
    return result


def ar2_predict(values, u):
    coefficients, u = parameters(values, 'cubic_ar2')['coefficient'], inputs(u)
    previous = older = previous_input = np.float32(0)
    result = np.empty(len(u), dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        for index, current in enumerate(u):
            row = np.array([previous, older, current, previous_input, previous * previous,
                            previous * previous * previous, 1], dtype=np.float32)
            prediction = np.sum(row * coefficients)
            result[index] = prediction
            older, previous, previous_input = previous, prediction, current
    return result


def predict(values, u, arm):
    if arm in PHASE:
        return phase_predict(values, u, arm)
    if arm == 'gru16':
        return gru_predict(values, u)
    require(arm == 'cubic_ar2', 'known recurrent method')
    return ar2_predict(values, u)


def ridge_design(u, y, kind):
    require(isinstance(u, np.ndarray) and u.dtype == np.float64 and u.ndim == 1
            and np.isfinite(u).all(), 'finite float64 ridge input')
    if kind == 'static_cubic':
        return np.column_stack((np.ones(len(u)), u, u * u, u * u * u)), 0
    if kind == 'cubic_ar2':
        require(isinstance(y, np.ndarray) and y.dtype == np.float64 and y.shape == u.shape
                and np.isfinite(y).all() and len(u) > 2, 'finite AR2 fit targets')
        previous = np.concatenate(([0.], y[:-1]))
        older = np.concatenate(([0., 0.], y[:-2]))
        previous_input = np.concatenate(([0.], u[:-1]))
        return np.column_stack((previous, older, u, previous_input, previous ** 2,
                                previous ** 3, np.ones(len(u)))), 2
    require(kind in ('fir128', 'fir512'), 'known ridge method')
    length = int(kind[3:])
    matrix = np.ones((len(u), length + 1), dtype=np.float64)
    for lag in range(length):
        matrix[:, lag + 1] = 0
        if lag < len(u):
            matrix[lag:, lag + 1] = u[:len(u) - lag]
    return matrix, length - 1


def fit_ridge(u, y, kind):
    design, skip = ridge_design(u, y, kind)
    require(y.shape == u.shape and y.dtype == np.float64 and np.isfinite(y).all()
            and len(u) > skip, 'complete ridge targets')
    design = design[skip:]
    precision = design.T @ design + RIDGE * np.eye(design.shape[1])
    information = design.T @ y[skip:]
    return cho_solve(cho_factor(precision, lower=True, check_finite=True), information,
                     check_finite=True)


def ridge_certificate(coefficient, independent, u, y, kind):
    """Backward error of the saved solve plus independent FIT design predictions.

    Coefficients can differ along poorly observed directions. A residual is
    therefore checked relative to the normal equation's own scale. The guard
    uses a fixed dimensional roundoff allowance, not an observed-data cutoff.
    AR2 FIT design predictions use legitimate past FIT outputs, as does its
    supervised ridge initialization; DEV checks below are fully free running.
    """
    design, skip = ridge_design(u, y, kind)
    dimension = design.shape[1]
    require(coefficient.dtype == independent.dtype == np.float64
            and coefficient.shape == independent.shape == (dimension,)
            and np.isfinite(coefficient).all() and np.isfinite(independent).all(), 'finite ridge solutions')
    used = design[skip:]
    matrix = used.T @ used + RIDGE * np.eye(dimension)
    information = used.T @ y[skip:]
    matrix_norm = float(np.linalg.norm(matrix, ord=np.inf))
    rhs_norm = float(np.linalg.norm(information, ord=np.inf))
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma = dimension * unit_roundoff / (1 - dimension * unit_roundoff)
    limit = 64 * gamma
    checks = {}
    for name, value in (('saved', coefficient), ('independent', independent)):
        residual = float(np.linalg.norm(matrix @ value - information, ord=np.inf))
        denominator = matrix_norm * float(np.linalg.norm(value, ord=np.inf)) + rhs_norm
        backward = residual / denominator if denominator > 0 else (0. if residual == 0 else math.inf)
        require(math.isfinite(backward) and backward <= limit, 'regularized normal-equation backward error')
        checks[name] = {'residual_inf': residual, 'denominator': denominator, 'backward_error': backward}
    saved_prediction, independent_prediction = design @ coefficient, design @ independent
    array_agreement(saved_prediction, independent_prediction, 'independent FIT ridge predictions')
    return {'dimension': dimension, 'ridge': RIDGE, 'unit_roundoff': unit_roundoff, 'gamma_d': gamma,
            'backward_error_limit': limit, 'normal_matrix_inf': matrix_norm, 'information_inf': rhs_norm,
            'solutions': checks, 'coefficient_max_abs_difference': float(np.max(np.abs(coefficient - independent))),
            'fit_prediction_max_abs_difference': float(np.max(np.abs(saved_prediction - independent_prediction))),
            'fit_prediction_rows': len(u), 'fitting_rows': len(used), 'dev_prediction_max_abs_difference': {}}


def classical_predict(coefficient, u, kind):
    design, _ = ridge_design(u, None, kind)
    require(isinstance(coefficient, np.ndarray) and coefficient.dtype == np.float64
            and coefficient.shape == (design.shape[1],) and np.isfinite(coefficient).all(),
            'finite classical coefficients')
    return design @ coefficient


def metric(prediction, target, y_std, *, skip=512):
    require(prediction.shape == target.shape and prediction.ndim == 1 and len(target) > skip
            and np.isfinite(target).all() and math.isfinite(y_std) and y_std > 0, 'metric inputs')
    finite = bool(np.isfinite(prediction).all())
    if not finite:
        return {'finite': False, 'rmse': None, 'nrmse': None}
    rmse = float(np.sqrt(np.mean((prediction[skip:].astype(np.float64) - target[skip:]) ** 2)))
    return {'finite': True, 'rmse': rmse, 'nrmse': rmse / y_std}


def compare(actual, expected, label, *, rtol=1e-10, atol=1e-12):
    if isinstance(expected, dict):
        require(type(actual) is dict and set(actual) == set(expected), label + ' keys')
        for key, value in expected.items():
            compare(actual[key], value, label + '/' + key, rtol=rtol, atol=atol)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), label + ' list')
        for index, value in enumerate(expected):
            compare(actual[index], value, label + '/' + str(index), rtol=rtol, atol=atol)
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ' value')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ' exact')


def parse_csv(raw):
    """Convert only the three frozen development intervals, never test values."""
    import csv
    import io
    import re

    require(type(raw) is bytes and bool(raw), 'nonempty CSV bytes')
    bounds = {'fit': (40650, 83946), 'dev_a': (84446, 92638), 'dev_b': (93138, 101330)}
    decimal = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
    result = {name: {'raw_indices': [], 'u': [], 'y': []} for name in bounds}
    try:
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
        require(next(reader) == ['V1', 'V2', ''], 'CSV header')
        row_index, blanks = 0, False
        for fields in reader:
            if not any(value.strip() for value in fields):
                blanks = True
                continue
            require(not blanks and len(fields) == 3 and not fields[2].strip(), 'CSV row geometry')
            for name, (start, end) in bounds.items():
                if start <= row_index < end:
                    result[name]['raw_indices'].append(row_index)
                    for key, text in zip(('u', 'y'), fields[:2], strict=True):
                        text = text.strip()
                        require(decimal.fullmatch(text) is not None, 'selected finite decimal field')
                        number = float(text)
                        require(math.isfinite(number), 'selected finite decimal field')
                        result[name][key].append(number)
            row_index += 1
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError('invalid CSV encoding or structure') from error
    require(row_index == 131072, 'complete raw row roster')
    return {name: {key: np.asarray(value, dtype=np.int64 if key == 'raw_indices' else np.float64)
                   for key, value in arrays.items()} for name, arrays in result.items()}


def array_agreement(actual, expected, label, *, exact=False, rtol=RTOL, atol=ATOL):
    require(isinstance(actual, np.ndarray) and actual.shape == expected.shape and actual.dtype == expected.dtype,
            label + ' shape/dtype')
    if exact or expected.dtype.kind not in 'fc':
        require(actual.tobytes() == expected.tobytes(), label + ' exact bytes')
    else:
        require(np.isfinite(actual).all() and np.isfinite(expected).all()
                and np.allclose(actual, expected, rtol=rtol, atol=atol), label + ' finite numerical agreement')


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary evidence file')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    descriptor(path)
    return json.loads(Path(path).read_text())


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == set(keys), 'exact NPZ keys: ' + path.name)
        return {key: values[key].copy() for key in keys}


CSV_SHA = 'ae62d5a91230c10f76e6dd02c8a4fac3c9d4d8a95fbf50e87cb0c4885003e0f1'
ARMS = ('fixed_phase', 'energy_phase', 'nonlinear_readout', 'gru16', 'cubic_ar2')
REFERENCES = ('static_cubic', 'fir128', 'fir512', 'cubic_ar2_frozen')
SOURCES = {'src/openjev/research/phase_memory.py', 'src/openjev/research/silverbox_data.py',
           'scripts/phase_study.py', 'scripts/audit_phase_study.py', 'tests/test_phase_memory.py',
           'tests/test_silverbox_data.py', 'tests/test_phase_study.py', 'tests/test_audit_phase_study.py',
           'research/phase-protocol.md'}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
QUAL_ARGV = ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_phase_memory.py',
             'tests/test_silverbox_data.py', 'tests/test_phase_study.py', 'tests/test_audit_phase_study.py']


def config():
    return {'version': 'phase-study-v1', 'seeds': list(SEEDS), 'arms': list(ARMS), 'references': list(REFERENCES),
            'updates': 2048, 'batch_size': 32, 'sequence_length': 256, 'train_burn': 64,
            'evaluation_burn': 512, 'window_seed_offset': 410000, 'ridge': 1e-6,
            'adam_lr': .003, 'ar2_adam_lr': .0001, 'adam_betas': [.9, .999], 'adam_eps': 1e-8,
            'gradient_clip': 1., 'wall_cap_seconds': 2400, 'fit_cap_seconds': 600,
            'partitions': {'fit': [40650, 83946], 'dev_a': [84446, 92638], 'dev_b': [93138, 101330]},
            'official_test_access': False, 'simulation_dtype': 'float32', 'ridge_dtype': 'float64',
            'prediction_storage_dtype': 'float64', 'scored_units': 'mV',
            'min_control_reduction': .1, 'baseline_noninferiority_ratio': 1.05,
            'competence_fit_std_fraction': .1}


def roster():
    names = {'registration.json', 'qualification.json', 'normalization.json', 'references.npz',
             'results.json', 'resources.json', 'run.json'}
    names.update('sources/' + name for name in SOURCES)
    names.update('data-' + name + '.npz' for name in ('fit', *PARTITIONS))
    for seed in SEEDS:
        names.add(f'windows-seed{seed}.npy')
        for arm in ARMS:
            names.update(f'models/{arm}/{seed}/{name}' for name in
                         ('initial.npz', 'final.npz', 'training.npz', 'optimizer.npz', 'fit.json'))
            names.update(f'prediction-{arm}-{seed}-{part}.npy' for part in PARTITIONS)
    names.update(f'prediction-{name}-{part}.npy' for name in REFERENCES for part in PARTITIONS)
    return names


def absolute(value):
    require(type(value) is str, 'string command arguments')
    path = Path(value)
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def closed_receipt(path, source_pins, *, cap, expected_argv=None):
    path = Path(path)
    receipt = read(path)
    require(set(receipt) == {'state', 'returncode', 'argv', 'elapsed_seconds', 'thread_env',
            'sources_before', 'sources_after', 'log_path', 'log', 'timeout_seconds', 'wrapper'},
            'complete original process receipt')
    require(receipt['state'] == 'EXITED' and type(receipt['returncode']) is int and receipt['returncode'] == 0,
            'original successful process closure')
    seconds = receipt['elapsed_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= cap
            and type(receipt['timeout_seconds']) is int and receipt['timeout_seconds'] == cap,
            'original process cap')
    require(receipt['sources_before'] == receipt['sources_after'] == source_pins,
            'original process source pins')
    require(receipt['thread_env'] == {key: '1' for key in THREADS}, 'original single-thread scope')
    require(type(receipt['argv']) is list and all(type(value) is str for value in receipt['argv']), 'string argv')
    if expected_argv is not None:
        require(receipt['argv'] == expected_argv, 'exact original argv')
    name = receipt['log_path']
    require(type(name) is str and bool(name) and Path(name).name == name, 'one sibling original log')
    require(descriptor(path.parent / name) == receipt['log'], 'original log hash')
    require(descriptor(ROOT / 'output/phase-engineering-v1/invoke.py') == receipt['wrapper'], 'original wrapper hash')
    return receipt


def authenticate(study, csv_path, run_receipt):
    """Metadata and opaque byte checks only; no CSV numeric conversion or NPZ decoding."""
    import re
    from importlib.metadata import version

    folder, csv_path = Path(study).resolve(), Path(csv_path).resolve()
    plan = read(folder / 'registration.json')
    require(set(plan) == {'config', 'csv', 'qualification', 'sources', 'environment'}, 'registration schema')
    compare(plan['config'], config(), 'fixed configuration', rtol=0, atol=0)
    require(set(plan['sources']) == SOURCES, 'all nine original sources')
    for name, pin in plan['sources'].items():
        require(descriptor(ROOT / name) == pin == descriptor(folder / 'sources' / name), 'source/snapshot bytes')
    require(descriptor(csv_path) == plan['csv'] and plan['csv']['sha256'] == CSV_SHA, 'official CSV opaque pin')
    expected_environment = {'python': sys.version, 'numpy': np.__version__, 'torch': version('torch'),
                            'threads': {key: '1' for key in THREADS}, 'device': 'cpu'}
    compare(plan['environment'], expected_environment, 'frozen runtime')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread audit runtime')
    manifest = read(folder / 'manifest.json')
    require(set(manifest) == {'files'} and set(manifest['files']) == roster(), 'complete original artifact roster')
    require(not any(path.is_symlink() for path in folder.rglob('*')), 'ordinary evidence paths')
    actual = {path.relative_to(folder).as_posix(): descriptor(path) for path in folder.rglob('*')
              if path.is_file() and path != folder / 'manifest.json'}
    require(actual == manifest['files'], 'complete unchanged opaque inventory')
    process = closed_receipt(run_receipt, plan['sources'], cap=2400)
    command = process['argv']
    require(len(command) == 11 and absolute(command[0]) == ROOT / '.venv/bin/python'
            and absolute(command[1]) == ROOT / 'scripts/phase_study.py'
            and command[2:4] == ['run', '--csv'] and absolute(command[4]) == csv_path
            and command[5] == '--registration' and command[7] == '--qualification'
            and command[9] == '--output' and absolute(command[10]) == folder, 'original run argv')
    registration_path, qualification_path = absolute(command[6]), absolute(command[8])
    require(descriptor(registration_path) == descriptor(folder / 'registration.json'), 'original registration join')
    require(descriptor(qualification_path) == plan['qualification'] == descriptor(folder / 'qualification.json'),
            'original qualification join')
    qualified = closed_receipt(qualification_path, plan['sources'], cap=300, expected_argv=QUAL_ARGV)
    require(qualified == read(folder / 'qualification.json'), 'copied qualification identity')
    run = read(folder / 'run.json')
    require(set(run) == {'version', 'registration', 'registration_commit', 'csv', 'models_fitted', 'ridge_fits',
        'optimizer_updates', 'fit_seconds', 'elapsed_seconds', 'numeric_partitions_loaded',
        'official_test_values_read', 'source_pins_verified_before_and_after'}, 'original run schema')
    fixed = {'version': 'phase-study-v1', 'registration': descriptor(folder / 'registration.json'),
             'csv': plan['csv'], 'models_fitted': 15, 'ridge_fits': 4, 'optimizer_updates': 15 * config()['updates'],
             'numeric_partitions_loaded': ['fit', 'dev_a', 'dev_b'], 'official_test_values_read': 0,
             'source_pins_verified_before_and_after': True}
    compare({key: run[key] for key in fixed}, fixed, 'original complete execution')
    require(type(run['elapsed_seconds']) in (float, int) and math.isfinite(run['elapsed_seconds'])
            and 0 < run['elapsed_seconds'] <= process['elapsed_seconds'], 'inner/outer timing join')
    require(set(run['fit_seconds']) == {f'{arm}/{seed}' for arm in ARMS for seed in SEEDS}, 'all fit times')
    require(all(type(value) in (float, int) and math.isfinite(value) and 0 < value <= 600
                for value in run['fit_seconds'].values()), 'fixed individual fit cap')
    require(sum(run['fit_seconds'].values()) <= run['elapsed_seconds'], 'fit durations within original process')
    commit = run['registration_commit']
    require(type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'pre-fit Git identity')
    for name, pin in {registration_path.relative_to(ROOT).as_posix(): run['registration'], **plan['sources']}.items():
        raw = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == pin, 'committed before fitting')
    return {'registration': descriptor(folder / 'registration.json'), 'manifest': descriptor(folder / 'manifest.json'),
            'run_receipt': descriptor(run_receipt), 'qualification': plan['qualification'], 'csv': plan['csv'],
            'sources': plan['sources'], 'registration_commit': commit, 'original_process_seconds': process['elapsed_seconds'],
            'original_inner_seconds': run['elapsed_seconds'], 'manifest_files': len(actual)}


def scores_for(prediction, normalized_target, y_std):
    require(prediction.dtype == np.float64 and prediction.shape == normalized_target.shape
            and prediction.shape == (8192,) and np.isfinite(prediction).all(), 'complete finite free-run output')
    error_mv = (prediction[512:] - normalized_target[512:]) * y_std * 1000
    return {'rmse_mv': float(np.sqrt(np.mean(error_mv * error_mv))),
            'mae_mv': float(np.mean(np.abs(error_mv))), 'scored_rows': 7680}


def evaluate_rule(scores, y_std):
    require(set(scores) == set(PARTITIONS) and math.isfinite(y_std) and y_std > 0, 'both scored partitions')
    conditions, summary = [{'name': 'all_completed_predictions_finite', 'passed': True}], {}
    keys = {f'{arm}/{seed}' for arm in ARMS for seed in SEEDS} | set(REFERENCES)
    for partition in PARTITIONS:
        rows = scores[partition]
        require(set(rows) == keys, 'all trained seeds and references')
        for row in rows.values():
            require(set(row) == {'rmse_mv', 'mae_mv', 'scored_rows'} and type(row['scored_rows']) is int
                    and row['scored_rows'] == 7680 and all(type(row[key]) in (int, float)
                    and math.isfinite(row[key]) and row[key] >= 0 for key in ('rmse_mv', 'mae_mv')), 'finite score row')
        means = {arm: sum(rows[f'{arm}/{seed}']['rmse_mv'] for seed in SEEDS) / len(SEEDS) for arm in ARMS}
        references = {name: rows[name]['rmse_mv'] for name in REFERENCES}
        best = min(means['gru16'], means['cubic_ar2'], *references.values())
        summary[partition] = {'family_mean_rmse_mv': means, 'reference_rmse_mv': references,
                              'best_conventional_rmse_mv': best}
        for control in ('fixed_phase', 'nonlinear_readout'):
            conditions.append({'name': f'{partition}_mean_energy_improves_{control}_10pct',
                               'passed': bool(means['energy_phase'] <= .9 * means[control])})
            conditions.extend({'name': f'{partition}_seed{seed}_energy_not_worse_{control}',
                               'passed': bool(rows[f'energy_phase/{seed}']['rmse_mv'] <= rows[f'{control}/{seed}']['rmse_mv'])}
                              for seed in SEEDS)
        conditions += [{'name': f'{partition}_within_5pct_best_conventional',
                        'passed': bool(means['energy_phase'] <= 1.05 * best)},
                       {'name': f'{partition}_competent_below_10pct_fit_std',
                        'passed': bool(means['energy_phase'] <= .1 * y_std * 1000)}]
    passed = sum(row['passed'] for row in conditions)
    return {'version': 'phase-study-v1', 'scope': 'Silverbox internal development simulation; official TEST unused',
            'scores': scores, 'summary': summary, 'conditions': conditions, 'passed': passed,
            'total': len(conditions), 'outcome': 'ADVANCE_PHASE_MECHANISM' if passed == len(conditions)
            else 'DO_NOT_ADVANCE_PHASE_MECHANISM'}


def validate_optimizer(values, arm, updates):
    expected = {f'{name}.{field}' for name in shapes(arm) for field in ('exp_avg', 'exp_avg_sq', 'step')}
    require(set(values) == expected, 'complete named Adam state')
    for name, shape in shapes(arm).items():
        for field in ('exp_avg', 'exp_avg_sq'):
            array = values[f'{name}.{field}']
            require(array.dtype == np.float32 and array.shape == shape and np.isfinite(array).all(), 'finite Adam moments')
        require((values[f'{name}.exp_avg_sq'] >= 0).all(), 'nonnegative Adam second moment')
        step = values[f'{name}.step']
        require(step.dtype == np.float32 and step.shape == () and float(step) == updates, 'exact completed Adam steps')


def validate_training(trace, fit, *, arm, seed, seconds):
    updates = config()['updates']
    require(set(trace) == {'loss', 'gradnorm'}, 'complete training trace')
    for name in trace:
        require(trace[name].dtype == np.float64 and trace[name].shape == (updates,)
                and np.isfinite(trace[name]).all() and (trace[name] >= 0).all(), 'finite complete training trace')
    compare(fit, {'arm': arm, 'seed': seed, 'updates': updates, 'elapsed_seconds': seconds,
             'loss_initial_batch': float(trace['loss'][0]), 'loss_final_batch': float(trace['loss'][-1]),
             'state_initialization': 'zero every training sequence', 'dev_or_test_access': False}, 'fit/trace identity')


def resources():
    trained = {}
    for arm in ARMS:
        count = sum(math.prod(shape) for shape in shapes(arm).values())
        state = 4 if arm in PHASE else (16 if arm == 'gru16' else 3)
        for seed in SEEDS:
            trained[f'{arm}/{seed}'] = {'parameters': count, 'parameter_bytes': count * 4,
               'state_scalars': state, 'state_bytes': state * 4, 'normalization_bytes': 32,
               'logical_total_bytes': count * 4 + state * 4 + 32}
    reference = {}
    for name, count, state_bytes in zip(REFERENCES, (4, 129, 513, 7), (0, 127 * 8, 511 * 8, 12), strict=True):
        parameter_bytes = count * (4 if name == 'cubic_ar2_frozen' else 8)
        reference[name] = {'coefficients': count, 'saved_coefficient_bytes': count * 8,
            'parameter_bytes': parameter_bytes, 'state_bytes': state_bytes,
            'normalization_bytes': 32, 'logical_total_bytes': parameter_bytes + state_bytes + 32}
    return {'trained_models': trained, 'references': reference,
        'scope': 'Parameter/state footprint; no wall-time speedup or matched-training-compute claim',
        'excluded': 'Python/native workspace, optimizer and training/audit data; no hidden measurement archive at inference'}


def audit(study, *, csv_path, run_receipt, check=lambda: None):
    started = time.perf_counter()
    folder = Path(study).resolve()
    check()
    admission = authenticate(folder, csv_path, run_receipt)
    parsed = parse_csv(Path(csv_path).read_bytes())
    array_loads = npz_decodes = npy_decodes = 0
    for partition, arrays in parsed.items():
        saved = load_npz(folder / f'data-{partition}.npz', arrays)
        npz_decodes += 1; array_loads += len(saved)
        for key in saved:
            array_agreement(saved[key], arrays[key], 'raw selected CSV/' + partition + '/' + key, exact=True)
    fit_u, fit_y = parsed['fit']['u'], parsed['fit']['y']
    normalization = {'u_mean': float(fit_u.mean()), 'u_std': float(fit_u.std()),
                     'y_mean': float(fit_y.mean()), 'y_std': float(fit_y.std())}
    require(normalization['u_std'] > 0 and normalization['y_std'] > 0, 'nonconstant FIT-only normalizers')
    compare(read(folder / 'normalization.json'), normalization, 'FIT-only normalization')
    normalized_u = (fit_u - normalization['u_mean']) / normalization['u_std']
    normalized_y = (fit_y - normalization['y_mean']) / normalization['y_std']
    refs = load_npz(folder / 'references.npz', REFERENCES)
    npz_decodes += 1; array_loads += len(refs)
    independent_refs, ridge_certificates = {}, {}
    for name in REFERENCES:
        kind = 'cubic_ar2' if name == 'cubic_ar2_frozen' else name
        independent_refs[name] = fit_ridge(normalized_u, normalized_y, kind)
        ridge_certificates[name] = ridge_certificate(refs[name], independent_refs[name], normalized_u, normalized_y, kind)
    run = read(folder / 'run.json')
    finals, cfg = {}, config()
    for seed in SEEDS:
        check()
        windows = np.load(folder / f'windows-seed{seed}.npy', allow_pickle=False)
        npy_decodes += 1; array_loads += 1
        expected_windows = np.random.Generator(np.random.PCG64(seed + cfg['window_seed_offset'])).integers(
            0, len(fit_u) - cfg['sequence_length'] + 1, size=(cfg['updates'], cfg['batch_size']), dtype=np.int64)
        array_agreement(windows, expected_windows, 'complete shared training window order', exact=True)
        initials = {}
        for arm in ARMS:
            location = folder / 'models' / arm / str(seed)
            initial = load_npz(location / 'initial.npz', shapes(arm))
            final = load_npz(location / 'final.npz', shapes(arm))
            initials[arm], finals[arm, seed] = parameters(initial, arm), parameters(final, arm)
            opt_keys = {f'{name}.{field}' for name in shapes(arm) for field in ('exp_avg', 'exp_avg_sq', 'step')}
            optimizer = load_npz(location / 'optimizer.npz', opt_keys)
            trace = load_npz(location / 'training.npz', ('loss', 'gradnorm'))
            npz_decodes += 4; array_loads += len(initial) + len(final) + len(optimizer) + len(trace)
            validate_optimizer(optimizer, arm, cfg['updates'])
            validate_training(trace, read(location / 'fit.json'), arm=arm, seed=seed,
                              seconds=run['fit_seconds'][f'{arm}/{seed}'])
        for name in BASE_SHAPES:
            for arm in ('energy_phase', 'nonlinear_readout'):
                array_agreement(initials[arm][name], initials['fixed_phase'][name], 'paired phase initialization', exact=True)
        for name in ('phase_logits',):
            array_agreement(initials['energy_phase'][name], np.zeros(2, np.float32), 'zero initial phase', exact=True)
        array_agreement(initials['nonlinear_readout']['cubic_readout_weights'], np.zeros((2, 2), np.float32),
                        'zero cubic readout', exact=True)
        array_agreement(initials['cubic_ar2']['coefficient'], refs['cubic_ar2_frozen'].astype(np.float32),
                        'AR2 ridge initial value', exact=True)
    scores, maximum_error = {}, {}
    for partition in PARTITIONS:
        check()
        arrays = parsed[partition]
        u = (arrays['u'] - normalization['u_mean']) / normalization['u_std']
        y = (arrays['y'] - normalization['y_mean']) / normalization['y_std']
        scores[partition] = {}
        for arm in ARMS:
            for seed in SEEDS:
                key = f'{arm}/{seed}'
                predicted = predict(finals[arm, seed], u.astype(np.float32), arm).astype(np.float64)
                actual = np.load(folder / f'prediction-{arm}-{seed}-{partition}.npy', allow_pickle=False)
                npy_decodes += 1; array_loads += 1
                array_agreement(actual, predicted, 'independent recurrence/' + partition + '/' + key)
                maximum_error[f'{partition}/{key}'] = float(np.max(np.abs(actual - predicted)))
                scores[partition][key] = scores_for(actual, y, normalization['y_std'])
        for name in REFERENCES:
            if name == 'cubic_ar2_frozen':
                predicted = ar2_predict({'coefficient': refs[name].astype(np.float32)}, u.astype(np.float32)).astype(np.float64)
                independent_prediction = ar2_predict({'coefficient': independent_refs[name].astype(np.float32)},
                                                     u.astype(np.float32)).astype(np.float64)
            else:
                predicted = classical_predict(refs[name], u, name)
                independent_prediction = classical_predict(independent_refs[name], u, name)
            actual = np.load(folder / f'prediction-{name}-{partition}.npy', allow_pickle=False)
            npy_decodes += 1; array_loads += 1
            array_agreement(actual, predicted, 'independent reference/' + partition + '/' + name)
            array_agreement(actual, independent_prediction, 'independent ridge solution DEV/' + partition + '/' + name)
            ridge_certificates[name]['dev_prediction_max_abs_difference'][partition] = float(
                np.max(np.abs(actual - independent_prediction)))
            maximum_error[f'{partition}/{name}'] = float(np.max(np.abs(actual - predicted)))
            scores[partition][name] = scores_for(actual, y, normalization['y_std'])
    result = evaluate_rule(scores, normalization['y_std'])
    compare(read(folder / 'results.json'), result, 'independent metrics and all 21 gates')
    resource = resources()
    compare(read(folder / 'resources.json'), resource, 'independent parameter/state geometry')
    check()
    require(authenticate(folder, csv_path, run_receipt) == admission, 'original evidence unchanged after audit')
    return {'version': VERSION, 'agreement': True, 'admission': admission, 'results': result, 'resources': resource,
            'prediction_max_abs_error': maximum_error, 'ridge_certificates': ridge_certificates,
            'tolerance': {'normalized_rtol': RTOL, 'normalized_atol': ATOL,
            'ridge_backward_error': '64 * gamma_d, gamma_d = d*u/(1-d*u), u = float64 epsilon/2'},
            'counts': {'npz_decodes': npz_decodes, 'npy_decodes': npy_decodes,
            'array_loads': array_loads, 'checkpoint_decodes': 30, 'optimizer_decodes': 15, 'prediction_replays': 38,
            'independent_ridge_solves': 4, 'ridge_certificates': 4, 'ridge_solution_prediction_comparisons': 12,
            'window_order_reconstructions': 3,
            'official_test_values_read': 0, 'producer_calls': 0, 'torch_calls': 0, 'optimizer_updates': 0},
            'seconds': time.perf_counter() - started,
            'limitations': ['Internal development partitions only, not official benchmark TEST or a novelty result.',
                'Independent NumPy saved-checkpoint inference; training losses, Adam state and elapsed times are structural receipts, not optimization replay.',
                'Fixed per-point tolerance is an engineering guard, not a proof of incremental stability for every learned recurrence.',
                'Source order attests that all final fits precede DEV loading; the auditor does not replay execution chronology.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'output', 'run-receipt', 'csv'):
        parser.add_argument('--' + name, type=Path, required=True)
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
