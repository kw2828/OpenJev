# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent NumPy evidence audit, with no author/producer/solver imports.

Only authenticate() may run before original process closure. All numerical
reconstruction is explicitly downstream of that admission barrier.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REG = 'research/fsm-author-nllfr-registration.json'
REGISTRATION_SHA256 = 'b1202af6803c20e0f689eaadb35a1b3c93bf91a70ee11cf33a41806c4599190c'
PREFIT_COMMIT = '4f0258928b21b5e34e6385567deff56558f8fc1a'
FREEZE = 'output/fsm-author-engineering-v1/nllfr-prefit-freeze.json'
VERSION = 'fsm-author-nllfr-study-v1'
SEEDS = (9201, 9202, 9203)
FIT = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV') for r in range(3) for p in range(2))
DEV = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV') for r in range(3, 6) for p in range(2))
ALLOWED_RAW = ('u_100mV_train', 'y_100mV_train', 'u_200mV_train', 'y_200mV_train')
KNOWN_RAW = {f'{v}_{a}_{p}' for a in ('100mV', '200mV', '300mV') for p in ('train', 'test') for v in ('u', 'y')}
LINEAR = tuple(f'varx{p}-ridge{a:g}' for p in (32, 64, 96) for a in (1e-6, .001, .1))
NEURAL = ('affine_output_only-lr0.0001', 'affine_feedback-lr0.0001', 'tanh_output_only-lr0.001', 'tanh_feedback-lr0.0003')
CANDIDATE = NEURAL[-1]
BLA, NLLFR = 'author_bla28', 'author_nllfr28'
FAMILIES = (*LINEAR, 'native_varx', *NEURAL, 'folded_affine_feedback')
ORDERED = [(f, None) for f in (*LINEAR, 'native_varx')]+[(f, s) for f in (*NEURAL, 'folded_affine_feedback') for s in SEEDS]
BASE = ('A', 'B_u', 'C_y', 'D_yu', 'u_mean', 'u_std', 'y_mean', 'y_std', 'ts')
EXTRA = ('B_w', 'C_z', 'D_yw', 'D_zu', 'W0', 'b0', 'W1', 'b1', 'W2', 'b2')
NAMES = (*BASE, *EXTRA)
POLICY = {'iterations': 16, 'trials': 8, 'damping': .001, 'armijo': .0001,
          'gradient_tol': 1e-8, 'scale_floor': 1e-8, 'rcond': 1e-12}
TOLERANCES = {'objective': {'atol': 1e-7, 'rtol': 1e-8},
              'replay': {'atol': 1e-8, 'rtol': 1e-8},
              'context': {'atol': 1e-10, 'rtol': 1e-8},
              'preprocessing': {'atol': 1e-10, 'rtol': 1e-10}}
EXPERIMENT = {'order': 28, 'nz': 16, 'nw': 8, 'hidden_layers': 2, 'width': 64,
              'activation': 'relu', 'seed': 42, 'sigma': .0001, 'samples': 8192,
              'realizations': 6, 'periods': 2, 'fs': 6400., 'offset': 820,
              'frequency_bins': 4097, 'max_iter': 10000, 'rtol': .001, 'atol': .00001,
              'frequency_weighting': False, 'print_every': -1, 'outer_timeout_seconds': 18000,
              'rss_cap_bytes': 32*1024**3, 'pythonhashseed': '0'}
ENV = {**{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
       'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'True', 'PYTHONHASHSEED': '0',
       'PYTHONDONTWRITEBYTECODE': '1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('nonfinite JSON '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def pin(path):
    blob = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def descriptor(path):
    return {'path': str(Path(path).resolve()), **pin(path)}


def relative(value):
    require(isinstance(value, str) and not Path(value).is_absolute() and '\\' not in value
            and all(p not in ('', '.', '..') for p in value.split('/')), 'canonical relative path')
    return Path(value)


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), 'regular evidence directory')
    found = {}
    for p in sorted(folder.rglob('*')):
        require(not p.is_symlink(), 'symlink evidence')
        if p.is_file():
            found[str(p.relative_to(folder))] = pin(p)
    return found


def close(actual, expected, *, rtol=1e-10, atol=1e-12):
    if isinstance(expected, np.ndarray):
        actual = np.asarray(actual)
        require(actual.shape == expected.shape and np.isfinite(actual).all()
                and np.allclose(actual, expected, rtol=rtol, atol=atol), 'array disagreement')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'mapping schema disagreement')
        for k, v in expected.items():
            close(actual[k], v, rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'list schema disagreement')
        for a, b in zip(actual, expected, strict=True):
            close(a, b, rtol=rtol, atol=atol)
    elif isinstance(expected, (float, np.floating)):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), 'scalar disagreement')
    else:
        require(type(actual) is type(expected) and actual == expected, 'exact value disagreement')


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as source:
        require(len(source.files) == len(set(source.files)) and set(source.files) == set(keys), 'NPZ roster')
        return {k: source[k].copy(order='K') for k in keys}


def finite(value, shape, *, complex_value=False):
    require(isinstance(value, np.ndarray) and value.dtype == (np.complex128 if complex_value else np.float64)
            and value.shape == shape and np.isfinite(value).all(), 'finite numeric geometry')


def validate_model(m, *, production=True):
    require(isinstance(m, dict) and set(m) == set(NAMES), 'model roster')
    a = m['A']
    require(isinstance(a, np.ndarray) and a.ndim == 2 and len(a) > 0, 'state geometry')
    require(all(isinstance(m[k], np.ndarray) and m[k].ndim == 2 for k in ('C_z', 'B_w', 'W0')), 'network geometry')
    n, z, w, h = len(a), len(m['C_z']), m['B_w'].shape[-1], len(m['W0'])
    shapes = {'A': (n, n), 'B_u': (n, 3), 'C_y': (3, n), 'D_yu': (3, 3),
              'B_w': (n, w), 'C_z': (z, n), 'D_yw': (3, w), 'D_zu': (z, 3),
              'W0': (h, z), 'W1': (h, h), 'W2': (w, h), 'b0': (h,), 'b1': (h,), 'b2': (w,),
              'u_mean': (3,), 'u_std': (3,), 'y_mean': (3,), 'y_std': (3,), 'ts': ()}
    for key, shape in shapes.items():
        finite(m[key], shape)
    require(min(n, z, w, h) > 0 and (m['u_std'] > 0).all() and (m['y_std'] > 0).all()
            and float(m['ts']) == 1/6400, 'normalization/time')
    if production:
        require((n, z, w, h) == (28, 16, 8, 64), 'registered architecture')
        require(sum(m[k].size for k in (*BASE[:4], *EXTRA)) == 7473, 'trainable count')
    return n


def trajectory(m, u, initial, *, jacobian=False):
    """Own row-state recurrence and chain-rule sensitivity, output before update."""
    n = validate_model(m, production=False)
    require(isinstance(u, np.ndarray) and u.ndim == 3 and u.shape[0] > 0, 'input rank')
    batch, horizon = u.shape[:2]
    finite(u, (batch, horizon, 3)); finite(initial, (batch, n))
    x = initial.copy()
    y = np.empty((batch, horizon, 3), dtype=np.float64)
    derivative = np.empty((batch, horizon, 3, n)) if jacobian else None
    chain = np.tile(np.eye(n), (batch, 1, 1)) if jacobian else None
    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(horizon):
            z = x@m['C_z'].T+u[:, t]@m['D_zu'].T
            hidden0 = z@m['W0'].T+m['b0']
            hidden1 = np.maximum(hidden0, 0)@m['W1'].T+m['b1']
            w = np.maximum(hidden1, 0)@m['W2'].T+m['b2']
            y[:, t] = x@m['C_y'].T+u[:, t]@m['D_yu'].T+w@m['D_yw'].T
            x = x@m['A'].T+u[:, t]@m['B_u'].T+w@m['B_w'].T
            if jacobian:
                local0 = (hidden0 > 0)[:, :, None]*(m['W0']@m['C_z'])[None]
                local1 = (hidden1 > 0)[:, :, None]*(m['W1'][None]@local0)
                localw = m['W2'][None]@local1
                derivative[:, t] = (m['C_y'][None]+m['D_yw'][None]@localw)@chain
                chain = (m['A'][None]+m['B_w'][None]@localw)@chain
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise FloatingPointError('nonfinite independent trajectory')
    if jacobian and (not np.isfinite(derivative).all() or not np.isfinite(chain).all()):
        raise FloatingPointError('nonfinite independent sensitivity')
    return y, x, derivative


def linear_seed(m, y, u):
    n, steps = len(m['A']), u.shape[1]
    block, forced, design, rhs = m['C_y'].copy(), np.zeros((1, n)), [], []
    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(steps):
            design.append(block.copy())
            rhs.append(y[:, t+1]-forced@m['C_y'].T-u[:, t]@m['D_yu'].T)
            forced = forced@m['A'].T+u[:, t]@m['B_u'].T
            if t+1 < steps:
                block = block@m['A']
    matrix = np.concatenate(design)
    vector = np.stack(rhs, axis=1).reshape(1, -1).T
    require(np.isfinite(matrix).all() and np.isfinite(vector).all(), 'nonfinite linear seed system')
    x, _, rank, singular = np.linalg.lstsq(matrix, vector, rcond=1e-12)
    require(np.isfinite(x).all() and np.isfinite(singular).all(), 'nonfinite linear seed')
    return x.T.copy(), {'rank': int(rank), 'singular_values': singular, 'rcond': 1e-12,
                        'time': 'second observed output'}


def context_loss(prediction, target):
    with np.errstate(over='ignore', invalid='ignore'):
        r = (prediction-target).reshape(-1)/np.sqrt(target.size)
        loss = float(np.dot(r, r)/2)
    if not np.isfinite(r).all() or not math.isfinite(loss):
        raise FloatingPointError('nonfinite context objective')
    return loss, r


def solve_context(m, target, u, seed):
    """Independent literal 16-direction/8-trial scaled damped GN policy."""
    x = seed.copy()
    trace, accepted, calls, jac_calls, proposals = [], 0, 0, 0, 0
    status, first = 'ITERATION_CAP', None
    for iteration in range(16):
        prediction, _, sensitivity = trajectory(m, u, x, jacobian=True)
        calls += 1; jac_calls += 1
        loss, r = context_loss(prediction, target)
        if first is None:
            first = loss
        j = sensitivity.reshape(target.size, x.size)/np.sqrt(target.size)
        norms = np.linalg.norm(j, axis=0)
        if not np.isfinite(norms).all():
            raise FloatingPointError('nonfinite Jacobian column norm')
        row = {'iteration': iteration, 'objective': loss, 'trials': []}
        trace.append(row)
        if norms.max() == 0:
            status = 'ZERO_JACOBIAN'; break
        scale = np.maximum(norms, 1e-8*norms.max())
        k = j/scale
        g = k.T@r
        require(np.isfinite(g).all() and np.isfinite(k).all(), 'nonfinite scaled gradient')
        row['scaled_gradient_inf'] = float(np.max(abs(g)))
        singular = np.linalg.svd(k, compute_uv=False)
        row['scaled_jacobian_rank'] = int(np.sum(singular > 1e-12*singular[0]))
        if row['scaled_gradient_inf'] <= 1e-8*(1+np.linalg.norm(r)):
            status = 'GRADIENT_TOL'; break
        augmented = np.concatenate([k, np.sqrt(.001)*np.eye(x.size)], axis=0)
        p = np.linalg.lstsq(augmented, np.concatenate([-r, np.zeros(x.size)]), rcond=1e-12)[0]
        delta, slope = p/scale, float(np.dot(g, p))
        require(np.isfinite(delta).all() and math.isfinite(slope), 'nonfinite context direction')
        if slope >= 0:
            row['reason'] = 'nonnegative_directional_derivative'; status = 'STALLED'; break
        accepted_here = False
        for trial in range(8):
            alpha = 2.**(-trial)
            with np.errstate(over='ignore', invalid='ignore'):
                proposed = x+alpha*delta[None]
            detail = {'alpha': alpha, 'accepted': False, 'nonfinite': False}
            proposals += 1
            try:
                if not np.isfinite(proposed).all():
                    raise FloatingPointError('nonfinite proposal')
                calls += 1
                trial_prediction = trajectory(m, u, proposed)[0]
                trial_loss = context_loss(trial_prediction, target)[0]
                detail['objective'] = trial_loss
                if trial_loss < loss and trial_loss <= loss+.0001*alpha*slope:
                    x = proposed; accepted += 1; accepted_here = True; detail['accepted'] = True
            except FloatingPointError:
                detail['nonfinite'] = True
            row['trials'].append(detail)
            if accepted_here:
                break
        if not accepted_here:
            status = 'STALLED'; break
    prediction, final, derivative = trajectory(m, u, x, jacobian=True)
    calls += 1; jac_calls += 1
    last, _ = context_loss(prediction, target)
    require(first is not None and last <= first, 'context objective worsened')
    s = np.linalg.svd(derivative.reshape(target.size, x.size)/np.sqrt(target.size), compute_uv=False)
    require(np.isfinite(s).all(), 'nonfinite final Jacobian diagnostics')
    record = {'status': status, 'initial_objective': first, 'final_objective': last,
              'directions_considered': len(trace), 'accepted_steps': accepted,
              'trajectory_evaluations': calls, 'jacobian_evaluations': jac_calls,
              'trial_attempts': proposals, 'final_jacobian_rank': int(np.sum(s > 1e-12*s[0])),
              'final_jacobian_singular_values': s, 'trace': trace,
              'status_scope': 'bounded context solve; no optimality or forecasting-accuracy certificate'}
    return final, x, record


def replay_request(m, y_context, u_context, future_u):
    n = validate_model(m, production=False)
    require(y_context.ndim == 3 and y_context.shape[0] == 1 and y_context.shape[1] >= 2, 'one independent context')
    finite(y_context, (1, y_context.shape[1], 3)); finite(u_context, (1, y_context.shape[1]-1, 3))
    finite(future_u, (1, future_u.shape[1], 3))
    y, u = (y_context-m['y_mean'])/m['y_std'], (u_context-m['u_mean'])/m['u_std']
    seed, linear = linear_seed(m, y, u)
    forecast, solved, detail = solve_context(m, y[:, 1:], u, seed)
    detail['linear_rank'] = linear['rank']
    output, final, _ = trajectory(m, (future_u-m['u_mean'])/m['u_std'], forecast)
    physical = output*m['y_std']+m['y_mean']
    require(np.isfinite(physical).all(), 'nonfinite physical outputs')
    finite(final, (1, n))
    return physical, final, forecast, {'requests': [detail], 'linear_seed_states': seed,
        'solved_context_start_states': solved, 'linear_seed_diagnostics': [linear], 'policy': POLICY.copy()}


def read_raw(path, expected_sha):
    """Only four allowed members are decoded; other member headers stay unopened."""
    blob = Path(path).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected_sha, 'raw snapshot pin')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) <= KNOWN_RAW
                and set(ALLOWED_RAW) <= set(archive.files), 'raw archive member names')
        values = {k: archive[k].copy(order='K') for k in ALLOWED_RAW}
    for value in values.values():
        finite(value, (8192, 3, 6, 2))
    return values


def raw_fit_preprocessing(raw):
    fit = {v: np.concatenate([raw[f'{v}_{a}_train'][:, :, :3, :] for a in ('100mV', '200mV')], axis=2)
           for v in ('u', 'y')}
    result, norm = {}, {}
    for name, x in fit.items():
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        scale = x.std(axis=(0, 2, 3), keepdims=True)
        require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), 'FIT scales')
        norm[name+'_mean'], norm[name+'_std'] = mean.reshape(3), scale.reshape(3)
        standardized = (x-mean)/scale
        result['raw_fit_'+name] = x
        result[name] = standardized.mean(axis=3)
        result[name.upper()] = np.fft.rfft(standardized, axis=0).mean(axis=3)
    return result, norm


def periodic_x0(bla, spectrum, *, samples=8192, offset=820):
    n = len(bla['A'])
    finite(spectrum, (samples//2+1, 3, spectrum.shape[2]), complex_value=True)
    z = np.exp(2j*np.pi*np.arange(samples//2+1)/samples)
    # Solve the full original BLA at every bin, not a diagonal approximation.
    response = np.linalg.solve(z[:, None, None]*np.eye(n)-bla['A'],
                               np.broadcast_to(bla['B_u'], (len(z), n, 3)))
    states = np.fft.irfft(response@spectrum, n=samples, axis=0)
    require(np.isfinite(states).all(), 'nonfinite periodic BLA states')
    return states[-offset].copy()


def native_objective(m, u, target, x0, *, offset=820):
    length, _, realizations = u.shape
    require(0 < offset <= length and length % 2 == 0, 'period/offset geometry')
    finite(u, (length, 3, realizations))
    finite(target, (length//2+1, 3, realizations), complex_value=True)
    finite(x0, (len(m['A']), realizations))
    inputs = np.concatenate([u[-offset:], u]).transpose(2, 0, 1).copy()
    output = trajectory(m, inputs, x0.T.copy())[0][:, offset:]
    spectrum = np.fft.rfft(output.transpose(1, 2, 0), axis=0)
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(np.sum(abs(target-spectrum)**2)/((length//2+1)*realizations))
    require(math.isfinite(value), 'nonfinite native objective')
    return value


def metrics(prediction, target, scale):
    require(prediction.shape == target.shape and prediction.ndim == 3
            and prediction.shape[0] > 0 and prediction.shape[1] > 0, 'score geometry')
    finite(prediction, (*prediction.shape[:2], 3)); finite(target, prediction.shape); finite(scale, (3,))
    require((scale > 0).all(), 'positive score scales')
    with np.errstate(over='ignore', invalid='ignore'):
        square = (prediction-target)**2
        mse = float(square.mean()); channel = np.sqrt(square.mean(axis=(0, 1)))
        native = channel*scale
    require(np.isfinite(square).all() and np.isfinite(native).all() and math.isfinite(mse), 'metric overflow')
    return {'mse': mse, 'rmse': math.sqrt(mse), 'per_channel_rmse': channel.tolist(),
            'native_output_per_channel_rmse': native.tolist(), 'requests': len(prediction), 'horizon': prediction.shape[1]}


def physical_metrics(prediction, target, scale):
    """Physical subtraction precedes division, preserving producer overflow semantics."""
    require(all(isinstance(v, np.ndarray) and v.dtype == np.float64 for v in (prediction, target, scale)), 'physical score dtype')
    require(prediction.ndim == 3 and prediction.shape == target.shape and prediction.shape[-1] == 3
            and min(prediction.shape[:2]) > 0 and scale.shape == (3,)
            and np.isfinite(scale).all() and (scale > 0).all(), 'physical score geometry')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        squared = ((prediction-target)/scale)**2
        mse = float(np.mean(squared))
        channel = np.sqrt(np.mean(squared, axis=(0, 1)))
        native = channel*scale
    require(np.isfinite(prediction).all() and np.isfinite(target).all()
            and np.isfinite(squared).all() and np.isfinite(native).all() and math.isfinite(mse), 'metric overflow')
    return {'mse': mse, 'rmse': math.sqrt(mse), 'per_channel_rmse': channel.tolist(),
            'native_output_per_channel_rmse': native.tolist(), 'requests': len(prediction), 'horizon': prediction.shape[1]}


def decisions(old, bla_rows, new_rows, complete):
    require([(e['family'], e['seed']) for e in old] == ORDERED, '25 comparison slots')
    groups = {f: [e for e in old if e['family'] == f] for f in FAMILIES}
    groups[BLA] = [{'seed': None, 'rows': bla_rows}]
    groups[NLLFR] = [{'seed': None, 'rows': new_rows}] if complete else []
    means, record, seed = {}, {}, {}
    for family, members in groups.items():
        if not members or not all([r['record_id'] for r in e['rows']] == list(DEV) and all(
            r['status'] == 'complete' and type(r.get('rmse')) in (int, float)
            and math.isfinite(r['rmse']) and r['rmse'] >= 0 for r in e['rows']) for e in members):
            continue
        record[family] = {rid: float(np.mean([e['rows'][i]['rmse'] for e in members])) for i, rid in enumerate(DEV)}
        seed[family] = {str(e['seed']): float(np.mean([r['rmse'] for r in e['rows']])) for e in members}
        means[family] = float(np.mean(list(record[family].values())))
    strongest = min((f for f in means if f != CANDIDATE), key=lambda f: (means[f], f), default=None)
    gates = dict.fromkeys(('five_percent_below_strongest_control', 'every_seed_below_strongest_control',
        'no_record_over_two_percent_strongest_control', 'both_amplitudes_below_strongest_control'), False)
    if complete and BLA in means and NLLFR in means and CANDIDATE in means and strongest is not None:
        c, b = record[CANDIDATE], record[strongest]
        gates['five_percent_below_strongest_control'] = means[CANDIDATE] <= .95*means[strongest]
        gates['every_seed_below_strongest_control'] = all(seed[CANDIDATE][str(s)] < seed[strongest].get(str(s), seed[strongest].get('None')) for s in SEEDS)
        gates['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV)
        gates['both_amplitudes_below_strongest_control'] = all(np.mean([c[r] for r in DEV if r.startswith(a)]) < np.mean([b[r] for r in DEV if r.startswith(a)]) for a in ('100mV', '200mV'))
    return {'status': 'CONTINUE_REFERENCE_CHECK' if all(gates.values()) else 'DO_NOT_CONTINUE_REFERENCE_CHECK',
        'reference_complete': complete, 'candidate': CANDIDATE, 'strongest_control': strongest,
        'conditions': gates, 'passed': sum(gates.values()), 'total': 4, 'family_means': means,
        'per_record_means': record, 'per_seed_means': seed,
        'per_amplitude_means': {f: {a: float(np.mean([v for r, v in rec.items() if r.startswith(a)])) for a in ('100mV', '200mV')} for f, rec in record.items()},
        'seed_rule': 'Paired seed for stochastic controls; common deterministic score otherwise. No new candidate selection.'}


def _closed_process(plan, registration, phase, folder, process):
    terminal, launch = read(process), read(process.parent/'launch.json')
    script = 'fit_nllfr.py' if phase == 'fit' else 'evaluate_nllfr.py'
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'),
               str(ROOT/'research/fsm_author/scripts'/script), '--registration',
               str(registration), '--output', str(folder)]
    require(terminal['phase'] == phase and terminal['command'] == command
            and terminal['registration_sha256'] == pin(registration)['sha256'], 'original phase/command/registration')
    require(launch['status'] == 'running' and all(terminal[k] == v for k, v in launch.items() if k != 'status'), 'original launch join')
    require(terminal['status'] in ('completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed')
            and terminal['end_identity_matches'] is True and terminal['identity_error'] is None, 'closed original process with unchanged identities')
    not_spawned = terminal['status'] == 'supervisor_failed' and terminal.get('pid') is None
    require((terminal['observed_exit_code'] is None if not_spawned else
             type(terminal['observed_exit_code']) is int and type(terminal['pid']) is int)
            and type(terminal['elapsed_seconds']) in (int, float)
            and math.isfinite(terminal['elapsed_seconds']) and terminal['elapsed_seconds'] > 0, 'terminal process fields')
    require((terminal['status'] != 'completed' or terminal['observed_exit_code'] == 0)
            and (terminal['status'] != 'failed' or terminal['observed_exit_code'] != 0), 'terminal exit consistency')
    require(terminal['outcome'] == ('ORIGINAL_PROCESS_COMPLETE' if terminal['status'] == 'completed' else 'INCOMPLETE'), 'terminal outcome')
    require(terminal['timeout_seconds'] == (18000 if phase == 'fit' else 3600)
            and terminal['rss_cap_bytes'] == 32*1024**3 and terminal['environment'] == ENV, 'runtime/caps')
    require(type(terminal['peak_polled_child_rss_bytes']) is int and terminal['peak_polled_child_rss_bytes'] >= 0, 'RSS evidence')
    require(terminal['producer'] == descriptor(ROOT/'research/fsm_author/scripts'/script)
            and terminal['supervisor'] == descriptor(ROOT/'research/fsm_author/scripts/run_nllfr_study.py'), 'launcher source identity')
    log_path = process.parent/'process.log'
    require((pin(log_path)['sha256'] if log_path.exists() else None) == terminal['log_sha256']
            and (log_path.exists() or not_spawned), 'original process log')
    files = inventory(folder) if folder.exists() else {}
    require(files == terminal['artifacts'], 'exact original output inventory')
    if 'started.json' in files:
        started = read(folder/'started.json')
        require(started['pid'] == terminal['pid'], 'child pid join')
        if phase == 'fit':
            require(started['registration'] == descriptor(registration), 'fit start registration')
        else:
            require(started['registration_sha256'] == pin(registration)['sha256'], 'evaluation start registration')
    if phase == 'fit' and terminal['status'] == 'completed':
        for name, digest in plan['source_sha256'].items():
            require(files['source/'+name]['sha256'] == digest, 'frozen source snapshot')
    return terminal, files


def qualification_evidence(paths, plan):
    """Inspect pinned qualification descriptors/logs only, with no package import."""
    q = read(paths['producer_qualification'])
    definition = read(q['definition']['path'])
    require(q['definition'] == descriptor(q['definition']['path'])
            and q['sources_before'] == q['sources_after'] == definition['sources'], 'qualification source closure')
    for name, value in definition['sources'].items():
        require(value == descriptor(ROOT/relative(name))
                and plan['source_sha256'][name] == value['sha256'], 'qualification current source identity')
    require(len(q['commands']) == 2 and [r['command'] for r in q['commands']] == definition['commands'], 'qualification commands')
    for row in q['commands']:
        require(row['returncode'] == 0 and row['log'] == descriptor(row['log']['path'])
                and math.isfinite(row['seconds']) and row['seconds'] >= 0, 'qualification original log')
    review = read(paths['source_review'])
    require(review['qualification'] == plan['prerequisites']['producer_qualification']
            and review['reviewed_source_sha256'] == plan['source_sha256'], 'source peer-review identity')
    return {'definition': q['definition'], 'logs': [r['log'] for r in q['commands']]}


def frozen_commit_evidence(plan, registration, started_time_ns):
    freeze = read(ROOT/FREEZE)
    require(freeze['status'] == 'FROZEN_BEFORE_MEASURED_LAUNCH' and freeze['prefit_commit'] == PREFIT_COMMIT
            and freeze['registration_sha256'] == REGISTRATION_SHA256 and freeze['push_observed_exit_code'] == 0,
            'prefit freeze receipt')
    require(datetime.fromisoformat(freeze['created_utc']).timestamp()*1e9 < started_time_ns,
            'freeze must precede original launch')
    for name, expected in {**plan['source_sha256'], str(registration.relative_to(ROOT)): REGISTRATION_SHA256}.items():
        blob = subprocess.run(['git', 'show', PREFIT_COMMIT+':'+str(relative(name))], cwd=ROOT,
                              check=True, capture_output=True).stdout
        require(hashlib.sha256(blob).hexdigest() == expected, 'committed source/registration pin')
    return descriptor(ROOT/FREEZE)


def authenticate(study, process, evaluation_process=None):
    """Metadata/opaque hashes only. No NPZ header or payload is decoded here."""
    study, process = Path(study).resolve(), Path(process).resolve()
    registration = ROOT/REG
    require(isinstance(REGISTRATION_SHA256, str) and pin(registration)['sha256'] == REGISTRATION_SHA256,
            'unbound/changed registration')
    plan = read(registration)
    require(plan['version'] == VERSION and plan['experiment'] == EXPERIMENT
            and plan['audit_tolerances'] == TOLERANCES, 'frozen recipe/tolerances')
    require(study == ROOT/relative(plan['output']) and process == ROOT/relative(plan['process_directory'])/'process.json', 'registered original fit paths')
    require(plan['evaluation']['outer_timeout_seconds'] == 3600
            and plan['evaluation']['rss_cap_bytes'] == 32*1024**3, 'evaluation caps')
    for name, digest in plan['source_sha256'].items():
        require(pin(ROOT/relative(name))['sha256'] == digest, 'source drift')
    paths = {}
    for key, value in plan['prerequisites'].items():
        path = ROOT/relative(value['path'])
        require(pin(path)['sha256'] == value['sha256'], 'prerequisite drift: '+key)
        paths[key] = path
    needed = {'runtime_preflight', 'nllfr_runtime_receipt', 'nllfr_runtime_process',
              'nllfr_runtime_observation', 'nllfr_runtime_definition', 'nllfr_runtime_initialization',
              'nllfr_runtime_initial_export', 'bla_final_zip', 'bla_final_npz', 'bla_fit', 'bla_summary',
              'bla_registration', 'bla_process', 'bla_audit', 'bla_audit_process', 'producer_qualification',
              'source_review', 'common_normalizer', 'parent_summary', 'parent_evaluations', 'parent_audit',
              'parent_process', 'parent_closure', 'parent_registration'}
    require(needed <= set(paths), 'required prerequisite roster')
    for k in ('runtime_preflight', 'nllfr_runtime_receipt', 'nllfr_runtime_process', 'producer_qualification', 'source_review'):
        require(read(paths[k])['status'] == 'PASS', 'failed prerequisite: '+k)
    runtime, observed, definition = (read(paths[k]) for k in ('nllfr_runtime_process', 'nllfr_runtime_observation', 'nllfr_runtime_definition'))
    execution = read(paths['nllfr_runtime_receipt'])
    require(runtime['sources_unchanged'] is True and len(runtime['stages']) == 2
            and all(s['returncode'] == 0 for s in runtime['stages']), 'qualified runtime original closure')
    require(observed['observed_exit_code'] == 0 and observed['process'] == descriptor(paths['nllfr_runtime_process'])
            and observed['execution_receipt'] == descriptor(paths['nllfr_runtime_receipt']), 'qualified runtime observation')
    require(runtime['definition'] == execution['definition'] == descriptor(paths['nllfr_runtime_definition'])
            and definition['fixture']['trainable_scalars'] == 7473
            and execution['details']['fresh_process_initialization_equal'] is True, 'qualified initialization definition')
    require(runtime['files']['fresh-initialization.json'] == descriptor(paths['nllfr_runtime_initialization'])
            and runtime['files']['initial.npz'] == descriptor(paths['nllfr_runtime_initial_export']), 'qualified random-initialization digest')
    for name, value in definition['sources'].items():
        require(value == descriptor(ROOT/relative(name)), 'qualified synthetic source drift')
    for stage in runtime['stages']:
        require(stage['log'] == descriptor(stage['log']['path']), 'original runtime stage log')
    qualified = qualification_evidence(paths, plan)
    # Both independent previous audits must still admit the exact pinned payloads.
    old_audit, bla_audit = read(paths['parent_audit']), read(paths['bla_audit'])
    for prior in (old_audit, bla_audit):
        require(prior['status'] == 'PASS' and prior['agreement'] is True, 'previous independent audit')
    reference = ROOT/'output/fsm-linear-controls-study-v1'
    require(paths['parent_summary'] == reference/'summary.json' and paths['parent_evaluations'] == reference/'evaluations.json', 'fixed parent scalar identities')
    for name in ('summary.json', 'evaluations.json', 'closure.json'):
        key = {'summary.json': 'parent_summary', 'evaluations.json': 'parent_evaluations', 'closure.json': 'parent_closure'}[name]
        require(old_audit['inputs']['files'][name] == pin(paths[key]), 'parent scalar audit join')
    require(old_audit['inputs']['registration'] == pin(paths['parent_registration'])
            and old_audit['inputs']['process']['sha256'] == pin(paths['parent_process'])['sha256'], 'parent original audit lineage')
    parent_process, parent_closure = read(paths['parent_process']), read(paths['parent_closure'])
    require(parent_process['observed_exit_code'] == 0 and parent_closure['status'] == 'completed'
            and parent_process['summary_sha256'] == parent_closure['summary_sha256'] == pin(paths['parent_summary'])['sha256'], 'parent original closure')
    for name, value in old_audit['inputs']['files'].items():
        require(pin(reference/relative(name)) == value, 'parent audited payload drift')
    bla_folder = paths['bla_final_npz'].parent
    require(bla_audit['results']['reference_status'] == 'REFERENCE_COMPLETE'
            and bla_audit['inputs']['process'] == descriptor(paths['bla_process'])
            and bla_audit['inputs']['registration'] == descriptor(paths['bla_registration']), 'completed BLA audit identity')
    bp, bap = read(paths['bla_process']), read(paths['bla_audit_process'])
    require(bp['status'] == 'completed' and bp['observed_exit_code'] == 0 and bp['end_identity_matches'] is True
            and bap['observed_exit_code'] == 0 and bap['audit_sha256'] == pin(paths['bla_audit'])['sha256'], 'original BLA/audit closure')
    require(read(paths['bla_fit'])['status'] == 'complete' and read(paths['bla_summary'])['status'] == 'REFERENCE_COMPLETE', 'BLA completion')
    for name, value in bla_audit['inputs']['files'].items():
        require(pin(bla_folder/relative(name)) == value, 'BLA audited payload drift')
    for key, name in (('bla_final_zip', 'final.zip'), ('bla_fit', 'fit.json'), ('bla_summary', 'summary.json')):
        require(paths[key] == bla_folder/name, 'BLA named payload join')
    require(bla_audit['inputs']['reference_evaluations'] == descriptor(paths['parent_evaluations'])
            and bla_audit['inputs']['reference_audit'] == descriptor(paths['parent_audit']), 'comparison lineage through BLA')
    require(pin(ROOT/relative(plan['data_path']))['sha256'] == plan['data_sha256'], 'raw archive drift')
    terminal, files = _closed_process(plan, registration, 'fit', study, process)
    require(terminal['parent_fit_process'] is None, 'fit cannot resume parent process')
    freeze = frozen_commit_evidence(plan, registration, terminal['started_time_ns'])
    evaluation = ROOT/relative(plan['evaluation_output'])
    et, ef = None, {}
    if evaluation_process is not None:
        evaluation_process = Path(evaluation_process).resolve()
        require(evaluation_process == ROOT/relative(plan['evaluation_process_directory'])/'process.json', 'registered evaluator process')
        require(terminal['status'] == 'completed', 'evaluation after successful original fit only')
        et, ef = _closed_process(plan, registration, 'evaluate', evaluation, evaluation_process)
        require(et['parent_fit_process'] == descriptor(process), 'evaluator original fit closure join')
    else:
        require(not evaluation.exists(), 'existing evaluation requires original terminal receipt')
    inputs = {'registration': descriptor(registration), 'process': descriptor(process),
        'process_log': descriptor(process.parent/'process.log') if (process.parent/'process.log').exists() else None, 'files': files,
        'prefit_freeze': freeze, 'producer_qualification': qualified,
        'evaluation_study': str(evaluation), 'evaluation_process': descriptor(evaluation_process) if et else None,
        'evaluation_files': ef, 'prerequisites': {k: descriptor(v) for k, v in paths.items()},
        'reference_summary': descriptor(paths['parent_summary']), 'reference_evaluations': descriptor(paths['parent_evaluations']),
        'raw': descriptor(ROOT/relative(plan['data_path'])), 'source': descriptor(Path(__file__))}
    return plan, inputs, paths, terminal, et


def check_runtime(identity, plan, preflight):
    require(identity['status'] == 'PASS' and identity['sources'] == plan['source_sha256']
            and identity['prerequisites'] == plan['prerequisites'] and identity['environment'] == ENV, 'runtime source/environment record')
    require(identity['python'] == preflight['python'] and identity['executable'] == preflight['executable']
            and identity['versions'] == preflight['versions'] and identity['direct_url'] == preflight['upstream_direct_url']
            and identity['installed_source_matches'] == preflight['installed_source_matches'], 'qualified runtime/source metadata')


def validate_fit(fit, trace, initial_pin, final_pin):
    n = fit['iterations']
    require(type(n) is int and 1 <= n <= 10000 and type(fit['author_stop_flag']) is bool, 'optimizer counters')
    require(set(trace) == {'loss_history', 'iter_times', 'iter_count', 'author_stop_flag', 'wall_time'}, 'raw trace roster')
    for key in ('loss_history', 'iter_times'):
        finite(trace[key], (n,)); require((trace[key] >= 0).all(), 'trace nonnegative values')
    require(trace['iter_count'].shape == () and np.issubdtype(trace['iter_count'].dtype, np.integer)
            and int(trace['iter_count']) == n, 'raw iteration count')
    require(trace['author_stop_flag'].shape == () and trace['author_stop_flag'].dtype == np.bool_
            and bool(trace['author_stop_flag']) == fit['author_stop_flag'], 'raw author stop flag')
    finite(trace['wall_time'], ())
    close(fit['author_reported_seconds'], float(trace['wall_time']))
    require(fit['status'] == ('complete' if fit['author_stop_flag'] else 'iteration_cap_reached')
            and (fit['author_stop_flag'] or n == 10000), 'fit completion meaning')
    require(fit['initial'] == initial_pin and fit['final'] == final_pin
            and fit['trainable_scalars'] == 7473 and fit['discarded_vendor_warmup_steps'] == 1
            and fit['zip_roundtrip_all_numeric_arrays_bitwise_equal'] is True, 'fit identity/scope')
    for key in ('optimization_and_preservation_compile_inclusive_seconds', 'author_reported_seconds'):
        require(type(fit[key]) in (float, int) and math.isfinite(fit[key]) and fit[key] >= 0, 'fit timing')
    require(type(fit['peak_ru_maxrss_bytes']) is int and fit['peak_ru_maxrss_bytes'] >= 0, 'raw RSS evidence')


def _numeric_failure(error):
    return isinstance(error, (FloatingPointError, np.linalg.LinAlgError)) or (
        type(error) is ValueError and str(error) in {'nonfinite linear seed system', 'nonfinite linear seed',
        'nonfinite scaled gradient', 'nonfinite context direction', 'nonfinite final Jacobian diagnostics',
        'nonfinite physical outputs', 'metric overflow'})


def validate_timing(row):
    keys = ('request_ms', 'slice_ms', 'initializer_ms', 'rollout_ms')
    require(all(type(row.get(k)) in (float, int) and math.isfinite(row[k]) and row[k] >= 0 for k in keys)
            and row['request_ms'] > 0, 'finite positive request timing')
    close(row['request_ms'], sum(row[k] for k in keys[1:]), atol=1e-8)


def validate_costs(evaluation, arrays):
    samples = evaluation['timings']
    roster = [(r, s) for r in DEV for s in (0, 7936)]
    require(len(samples) <= 24 and [(r['record_id'], r['start']) for r in samples] == roster[:len(samples)], 'timing original attempted prefix')
    for row in samples:
        require(set(row) == {'record_id', 'start', 'request_ms', 'slice_ms', 'initializer_ms', 'rollout_ms'}, 'timing row fields')
        validate_timing(row)
    if evaluation['timing_error'] is None:
        require(len(samples) == 24, 'complete timing coverage')
        close(evaluation['median_request_ms'], float(np.median([v['request_ms'] for v in samples])))
    else:
        require(isinstance(evaluation['timing_error'], str) and evaluation['timing_error']
                and evaluation['median_request_ms'] is None, 'timing failure retained')
    numeric = sum(v.nbytes for v in arrays.values())+28*8+9*8
    require(evaluation['persistent_numeric_bytes'] == numeric, 'all deployed numeric storage')
    return numeric


def references(paths, common):
    """Re-score authenticated old banks, never reconstruct or execute old models."""
    old = read(paths['parent_evaluations'])
    require([(e['family'], e['seed']) for e in old] == ORDERED, 'comparison roster')
    folder = paths['parent_evaluations'].parent
    targets, count = {}, 0
    for e in old:
        require([r['record_id'] for r in e['rows']] == list(DEV), 'parent record roster')
        stem = e['family']+(f'-{e["seed"]}' if e['seed'] is not None else '')
        for row in e['rows']:
            if row['status'] != 'complete':
                require(row['status'] in ('failed', 'not_run'), 'parent explicit failure'); continue
            b = load_arrays(folder/stem/'evaluation'/(row['record_id']+'.npz'), ('prediction', 'target', 'starts'))
            require(b['starts'].dtype == np.int64 and np.array_equal(b['starts'], np.arange(0, 7937, 256)), 'old starts')
            close(row, {'record_id': row['record_id'], 'status': 'complete', **metrics(b['prediction'], b['target'], common['y_scale'])})
            target = targets.setdefault(row['record_id'], b['target'])
            require(np.array_equal(b['target'], target), 'old exact target agreement')
            count += 1
    bla_eval = read(paths['bla_final_npz'].parent/'evaluation.json')
    require([r['record_id'] for r in bla_eval['rows']] == list(DEV), 'BLA record roster')
    for row in bla_eval['rows']:
        require(row['status'] == 'complete', 'required closed complete BLA')
        bank = load_arrays(paths['bla_final_npz'].parent/'evaluation'/(row['record_id']+'.npz'),
            ('prediction', 'target', 'starts', 'final_state', 'singular_values', 'context_residual_norm', 'y_context', 'u_context', 'future_u'))
        require(np.array_equal(bank['target'], targets[row['record_id']])
                and np.array_equal(bank['starts'], np.arange(0, 7937, 256)), 'BLA target/geometry join')
        scored = metrics(bank['prediction'], bank['target'], common['y_scale'])
        close({k: row[k] for k in scored}, scored)
    return old, bla_eval['rows'], targets, count


def physical_inputs(raw, rid, start):
    amplitude, _, realization, _, period = rid.split('-')
    realization, period = int(realization), int(period)
    y = raw[f'y_{amplitude}_train'][:, :, realization, period]
    u = raw[f'u_{amplitude}_train'][:, :, realization, period]
    return (y[start:start+100][None].copy(), u[start+1:start+100][None].copy(),
            u[start+100:start+228][None].copy(), y[start+100:start+228][None].copy())


def evaluate_replay(final, evaluation_folder, inventory_before, raw, common, targets):
    from collections import Counter

    result = read(evaluation_folder/'evaluation.json')
    require([r['record_id'] for r in result['rows']] == list(DEV), 'twelve evaluation records')
    expected = {'started.json', 'fit-provenance.json', 'identity-before.json', 'identity-after.json',
                'admission.json', 'events.jsonl', 'evaluation.json', 'summary.json'}
    statuses, attempts, differences, all_rows = Counter(), [], {}, []
    replayed, failed = 0, 0
    for rid, saved_record in zip(DEV, result['rows'], strict=True):
        banks, successful = [], 0
        for index, start in enumerate(range(0, 7937, 256)):
            stem = f'evaluation/{rid}/{start:04d}'
            expected.update({stem+'.json', stem+'.inputs.npz'})
            row = read(evaluation_folder/(stem+'.json'))
            require(row['record_id'] == rid and type(row['start']) is int and row['start'] == start, 'attempt identity')
            y, u, future, physical_target = physical_inputs(raw, rid, start)
            captured = load_arrays(evaluation_folder/(stem+'.inputs.npz'), ('starts', 'y_context', 'u_context', 'future_u'))
            require(captured['starts'].dtype == np.int64 and np.array_equal(captured['starts'], [start]), 'attempt starts')
            for key, value in (('y_context', y), ('u_context', u), ('future_u', future)):
                require(np.array_equal(captured[key], value) and captured[key].dtype == np.float64, 'exact raw request lineage')
            # Inputs are exclusively arrived context + executed future inputs. Targets
            # are used for scoring only after replay returns.
            replay_error = None
            try:
                physical, state, forecast, diagnostic = replay_request(final, y, u, future)
                normalized = (physical-common['y_mean'])/common['y_scale']
                target = (physical_target-common['y_mean'])/common['y_scale']
                # Check the physical-difference score path too, for honest overflows.
                physical_metrics(physical, physical_target, common['y_scale'])
                scored = metrics(normalized, target, common['y_scale'])
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                if not _numeric_failure(error):
                    raise
                replay_error = error
            replayed += 1
            attempts.append({'record_id': rid, 'start': start, 'status': row['status']})
            if row['status'] == 'failed':
                require(set(row) == {'record_id', 'start', 'status', 'error', 'traceback'}
                        and isinstance(row['error'], str) and isinstance(row['traceback'], str), 'failed attempt evidence')
                require(replay_error is not None and stem+'.npz' not in inventory_before, 'failed request independently reproduced')
                failed += 1; continue
            require(row['status'] == 'complete' and replay_error is None, 'successful request independent agreement')
            expected.add(stem+'.npz')
            bank = load_arrays(evaluation_folder/(stem+'.npz'), ('prediction', 'target', 'starts',
                'final_state', 'forecast_state', 'linear_seed_states', 'solved_context_start_states',
                'y_context', 'u_context', 'future_u'))
            require(bank['starts'].dtype == np.int64 and np.array_equal(bank['starts'], [start]), 'saved successful starts')
            for key, value in (('y_context', y), ('u_context', u), ('future_u', future), ('target', target)):
                require(bank[key].dtype == np.float64 and np.array_equal(bank[key], value), 'exact successful request inputs/target')
            require(np.array_equal(bank['target'][0], targets[rid][index]), 'unchanged parent target')
            pairs = {'prediction': normalized, 'final_state': state, 'forecast_state': forecast,
                     'linear_seed_states': diagnostic['linear_seed_states'],
                     'solved_context_start_states': diagnostic['solved_context_start_states']}
            differences[stem] = {}
            for key, actual in pairs.items():
                finite(bank[key], actual.shape)
                close(bank[key], actual, **TOLERANCES['replay'])
                differences[stem][key] = float(np.max(abs(bank[key]-actual)))
            # Exact status, ranks, trial choices and work counts; only numeric
            # objectives/diagnostics receive the predeclared context tolerance.
            close(row['context'], diagnostic, **TOLERANCES['context'])
            close({k: row[k] for k in scored}, metrics(bank['prediction'], bank['target'], common['y_scale']))
            validate_timing(row)
            state_max = max(float(np.max(abs(bank[k]))) for k in pairs if k != 'prediction')
            close(row['retained_state_max_abs'], state_max)
            require(set(row) == {'record_id', 'start', 'status', *scored, 'request_ms', 'slice_ms',
                    'initializer_ms', 'rollout_ms', 'context', 'retained_state_max_abs'}, 'complete attempt schema')
            statuses[diagnostic['requests'][0]['status']] += 1
            successful += 1; banks.append(bank)
        if successful == 32:
            name = 'evaluation/'+rid+'.npz'; expected.add(name)
            merged = load_arrays(evaluation_folder/name, ('prediction', 'target', 'starts', 'final_state',
                'y_context', 'u_context', 'future_u'))
            require(merged['starts'].dtype == np.int64 and np.array_equal(merged['starts'], np.arange(0, 7937, 256)), 'merged starts')
            for key in merged.keys()-{'starts'}:
                require(np.array_equal(merged[key], np.concatenate([b[key] for b in banks])), 'record bank agrees with all32 attempts')
            computed = {'record_id': rid, 'status': 'complete', **metrics(merged['prediction'], merged['target'], common['y_scale'])}
        else:
            computed = {'record_id': rid, 'status': 'incomplete', 'completed_requests': successful, 'expected_requests': 32}
        close(saved_record, computed); all_rows.append(computed)
    require(result['requests'] == attempts and result['context_status_counts'] == dict(statuses), 'all384 identities/status counts')
    storage = validate_costs(result, final)
    require(set(inventory_before) == expected, 'exact evaluation file roster')
    return result, all_rows, differences, replayed, failed, storage


def _result(study, inputs, scientific, counts, **extra):
    return {'status': 'PASS', 'agreement': True, 'study': str(study), 'inputs': inputs,
            'scientific_status': scientific, 'counts': counts, 'tolerances': TOLERANCES,
            'scope': 'Independent NumPy preprocessing, equations, context solve and saved-array scoring. No author model/optimizer imports, gradients, fitting, timing replay or candidate reselection. Only four allowed TRAIN members decoded; no reserved members/headers.', **extra}


def audit(study, process, evaluation_process=None):
    study, process = Path(study).resolve(), Path(process).resolve()
    plan, inputs, paths, terminal, et = authenticate(study, process, evaluation_process)
    counts = {'original_fit_attempts': 1, 'raw_archive_decodes': 0, 'raw_members_decoded': 0,
              'native_objective_replays': 0, 'periodic_bla_solves': 0, 'request_replays': 0,
              'old_forecast_banks_rescored': 0, 'old_model_replays': 0, 'timing_replays': 0,
              'refits': 0, 'optimizer_calls': 0, 'backward_calls': 0}
    if terminal['status'] != 'completed':
        return _result(study, inputs, 'REFERENCE_INCOMPLETE', counts,
            results={'reference_status': 'REFERENCE_INCOMPLETE', 'continuation': None,
                     'original_stop': terminal['status']}, audit_scope='Metadata-only closed failed fit; partial artifacts retained without completion claim.')
    files = inputs['files']
    expected = {'started.json', 'identity-before.json', 'identity-after.json', 'admission.json', 'events.jsonl',
        'initial.zip', 'initial.npz', 'original-bla.npz', 'fit-data.npz', 'initialization.json', 'final.zip',
        'final.npz', 'solver-trace.npz', 'roundtrip.npz', 'roundtrip-original-bla.npz', 'serialization.json',
        'fit.json', 'summary.json', *('source/'+n for n in plan['source_sha256'])}
    require(set(files) == expected, 'exact completed fit inventory')
    preflight = read(paths['runtime_preflight'])
    identity = read(study/'identity-before.json')
    require(identity == read(study/'identity-after.json'), 'fit before/after identities')
    check_runtime(identity, plan, preflight)
    admission = read(study/'admission.json')
    require(admission['source_sha256'] == plan['data_sha256'] and admission['decoded_keys'] == list(ALLOWED_RAW)
            and admission['fit_ids'] == list(FIT) and admission['optimization_records'] == 12
            and admission['dev_evaluation_calls'] == 0 and admission['original_bla_zip'] == descriptor(paths['bla_final_zip']), 'FIT-only admission')
    # First numerical read occurs only after all original/source/prerequisite pins.
    initial, final = (load_arrays(study/(name+'.npz'), NAMES) for name in ('initial', 'final'))
    validate_model(initial); validate_model(final)
    parameter_changes = {k: {'changed_scalars': int(np.count_nonzero(initial[k] != final[k])),
                             'total_scalars': initial[k].size} for k in (*BASE[:4], *EXTRA)}
    bla = load_arrays(paths['bla_final_npz'], BASE)
    original = load_arrays(study/'original-bla.npz', BASE)
    for key in BASE:
        require(np.array_equal(bla[key], original[key]) and np.array_equal(initial[key], bla[key]), 'exact original BLA initialization')
    for key in BASE[4:]:
        require(np.array_equal(final[key], bla[key]), 'frozen normalizers/sample time')
    qualified = load_arrays(paths['nllfr_runtime_initial_export'], NAMES)
    validate_model(qualified)
    for key in EXTRA:
        require(np.array_equal(initial[key], qualified[key]), 'exact qualified random initialization')
    initialization = read(study/'initialization.json')
    actual_hashes = {k: {'shape': list(initial[k].shape), 'dtype': str(initial[k].dtype),
        'sha256': hashlib.sha256(initial[k].tobytes(order='C')).hexdigest()} for k in EXTRA}
    require(initialization['random_component_hashes'] == actual_hashes
            and initialization['trainable_scalars'] == 7473 and initialization['trainable_leaves'] == 14
            and initialization['pythonhashseed'] == '0' and initialization['sample_interval_type'] == 'float', 'initial optimizer/random metadata')
    init_receipt = read(paths['nllfr_runtime_initialization'])
    require(all(init_receipt['arrays'][k] == actual_hashes[k] for k in EXTRA), 'qualified initialization hashes')
    raw = read_raw(ROOT/relative(plan['data_path']), plan['data_sha256'])
    counts.update(raw_archive_decodes=1, raw_members_decoded=4)
    derived, norm = raw_fit_preprocessing(raw)
    saved = load_arrays(study/'fit-data.npz', ('raw_fit_u', 'raw_fit_y', 'u', 'y', 'U', 'Y', 'f_idx', 'training_x0', 'independent_x0'))
    for key, value in derived.items():
        finite(saved[key], value.shape, complex_value=key in ('U', 'Y'))
        if key.startswith('raw_'):
            require(np.array_equal(saved[key], value), 'exact raw FIT assignment')
        else:
            close(saved[key], value, **TOLERANCES['preprocessing'])
    for key, value in norm.items():
        close(bla[key], value, **TOLERANCES['preprocessing'])
    for k in ('u', 'y'):
        close(np.fft.rfft(saved[k], axis=0), saved[k.upper()], **TOLERANCES['preprocessing'])
    require(saved['f_idx'].dtype == np.int64 and np.array_equal(saved['f_idx'], np.arange(1, 3840)), 'BLA excited indices')
    x0 = periodic_x0(bla, derived['U'])
    counts['periodic_bla_solves'] = 4097
    close(saved['training_x0'], x0, **TOLERANCES['preprocessing'])
    close(saved['independent_x0'], x0, **TOLERANCES['preprocessing'])
    close(initialization['training_x0_max_absolute_error'], float(np.max(abs(saved['training_x0']-saved['independent_x0']))))
    objectives = [native_objective(m, derived['u'], derived['Y'], x0) for m in (initial, final)]
    counts['native_objective_replays'] = 2
    fit = read(study/'fit.json')
    trace = load_arrays(study/'solver-trace.npz', ('loss_history', 'iter_times', 'iter_count', 'author_stop_flag', 'wall_time'))
    validate_fit(fit, trace, descriptor(study/'initial.npz'), descriptor(study/'final.npz'))
    close(initialization['initial_fit_loss'], objectives[0], **TOLERANCES['objective'])
    close(fit['initial_fit_loss'], objectives[0], **TOLERANCES['objective'])
    close(fit['final_fit_loss'], objectives[1], **TOLERANCES['objective'])
    require(objectives[1] <= objectives[0], 'returned native objective worsening')
    for path, values, keys in (('roundtrip.npz', final, NAMES), ('roundtrip-original-bla.npz', bla, BASE)):
        roundtrip = load_arrays(study/path, keys)
        require(all(np.array_equal(roundtrip[k], values[k]) and roundtrip[k].dtype == values[k].dtype for k in keys), 'exact retained serialization roundtrip')
    serialization = read(study/'serialization.json')
    require(serialization['all_live_arrays_bitwise_equal'] is True and serialization['original_bla_arrays_bitwise_equal'] is True, 'serialization validation flags')
    require(serialization['files'] == {k: descriptor(study/k) for k in ('final.zip', 'final.npz', 'original-bla.npz', 'roundtrip.npz', 'roundtrip-original-bla.npz')}, 'serialization file joins')
    fit_summary = read(study/'summary.json')
    require(fit_summary['status'] == ('FIT_COMPLETE' if fit['status'] == 'complete' else 'FIT_INCOMPLETE')
            and fit_summary['fit'] == fit and fit_summary['dev_evaluation_calls'] == fit_summary['model_selection_calls'] == 0, 'fit summary')
    events = [json.loads(line) for line in (study/'events.jsonl').read_text().splitlines()]
    require([e['phase'] for e in events] == ['optimization_started', 'fit_closed']
            and events[0]['time_ns'] <= events[1]['time_ns'], 'fit chronology')
    close(events[0]['initial_fit_loss'], objectives[0], **TOLERANCES['objective'])
    require(events[0]['max_iter'] == 10000 and events[1]['iterations'] == fit['iterations']
            and events[1]['status'] == fit_summary['status'], 'original fit event counters')
    if et is None or et['status'] != 'completed':
        require(authenticate(study, process, evaluation_process)[1] == inputs, 'source/input closure')
        return _result(study, inputs, 'REFERENCE_INCOMPLETE', counts,
            results={'reference_status': 'REFERENCE_INCOMPLETE', 'fit_status': fit['status'], 'continuation': None},
            native_objectives=objectives, parameter_changes=parameter_changes, audit_scope='Completed fit reconstructed; evaluation absent or original evaluator incomplete, with all partial evidence retained.')
    evaluation_folder = Path(inputs['evaluation_study'])
    before = read(evaluation_folder/'identity-before.json')
    require(before == read(evaluation_folder/'identity-after.json'), 'evaluator before/after source identity')
    require(before['source_sha256'] == plan['source_sha256'] and before['prerequisites'] == plan['prerequisites']
            and before['python'] == preflight['python'] and before['executable'] == preflight['executable']
            and before['versions'] == preflight['versions'] and before['direct_url'] == preflight['upstream_direct_url']
            and before['installed_author_sha256'] == {k: v['sha256'] for k, v in preflight['installed_source_matches'].items()}
            and all(before['thread_environment'][k] == v for k, v in ENV.items()), 'evaluator qualified runtime')
    provenance = read(evaluation_folder/'fit-provenance.json')
    require(provenance == {'process_sha256': pin(process)['sha256'], 'process': terminal,
        'fit_sha256': pin(study/'fit.json')['sha256'], 'registration_sha256': REGISTRATION_SHA256}, 'evaluation closed-fit provenance')
    admission = read(evaluation_folder/'admission.json')
    require(admission['decoded_keys'] == list(ALLOWED_RAW) and admission['dev_ids'] == list(DEV), 'declared DEV decoding')
    common = load_arrays(paths['common_normalizer'], ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    for value in common.values():
        finite(value, (3,))
    for key in ('u', 'y'):
        joined = derived['raw_fit_'+key].transpose(2, 3, 0, 1).reshape(-1, 3)
        close(common[key+'_mean'], joined.mean(axis=0), **TOLERANCES['preprocessing'])
        close(common[key+'_scale'], joined.std(axis=0), **TOLERANCES['preprocessing'])
    old, bla_rows, targets, old_count = references(paths, common)
    counts['old_forecast_banks_rescored'] = old_count
    counts['bla_forecast_banks_rescored'] = 12
    evaluation, rows, differences, requests, failures, storage = evaluate_replay(final, evaluation_folder, inputs['evaluation_files'], raw, common, targets)
    complete = fit['status'] == 'complete' and failures == 0 and evaluation['median_request_ms'] is not None
    summary = read(evaluation_folder/'summary.json')
    scientific = 'REFERENCE_COMPLETE' if complete else 'REFERENCE_INCOMPLETE'
    require(summary['status'] == scientific and summary['fit_status'] == fit['status'], 'evaluation scientific completion')
    mean = float(np.mean([r['rmse'] for r in rows])) if failures == 0 else None
    close(summary['mean_rmse'], mean); close(summary['median_request_ms'], evaluation['median_request_ms'])
    require(summary['persistent_numeric_bytes'] == storage and summary['context_status_counts'] == evaluation['context_status_counts'], 'summary storage/context statuses')
    events = [json.loads(line) for line in (evaluation_folder/'events.jsonl').read_text().splitlines()]
    require([e['phase'] for e in events] == ['record_closed']*12+['evaluation_closed']
            and all(a['time_ns'] <= b['time_ns'] for a, b in pairwise(events)), 'evaluation chronology')
    for event, row in zip(events[:12], rows, strict=True):
        close({k: event[k] for k in row}, row)
    close({k: events[-1][k] for k in summary}, summary)
    counts.update(request_replays=requests, failed_request_replays=failures, record_rows=12,
                  timing_samples=len(evaluation['timings']), complete_request_banks=requests-failures)
    require(authenticate(study, process, evaluation_process)[1] == inputs, 'source/input/output pins unchanged at audit close')
    return _result(study, inputs, scientific, counts, results={'reference_status': scientific,
        'fit_status': fit['status'], 'rows': rows, 'reference_mean_rmse': mean,
        'median_request_ms': evaluation['median_request_ms'], 'persistent_numeric_bytes': storage,
        'context_status_counts': evaluation['context_status_counts'],
        'continuation': decisions(old, bla_rows, rows, complete)}, native_objectives=objectives,
        replay_differences=differences, parameter_changes=parameter_changes, serialization_scope='Numeric roundtrip exports independently equal; ZIP deserialization itself was executed by the pinned producer, not this auditor.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--process', type=Path, required=True)
    parser.add_argument('--evaluation-process', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive audit output')
    result = audit(args.study, args.process, args.evaluation_process)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'scientific_status': result['scientific_status'], 'counts': result['counts']}))


if __name__ == '__main__':
    main()
