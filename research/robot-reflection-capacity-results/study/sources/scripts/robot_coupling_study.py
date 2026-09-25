"""Predeclared conditional multi-joint forecasting; official TEST stays closed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from openjev.research.industrial_robot_data import PARTITIONS, PREPROCESSING_SHA256, load_recording
from openjev.research.joint_coupling import ARMS as GRAPH_ARMS
from openjev.research.joint_coupling import JointCoupling

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-coupling-study-v1'
ARMS = (*GRAPH_ARMS, 'gru_residual', 'quadratic_ar2')
SEEDS = (8101, 8102, 8103)
RATES = (.0001, .001)
REFERENCES = ('linear_frozen', 'quadratic_frozen', 'persistence', 'velocity', 'direct_ridge_1', 'direct_ridge_100')
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py',
           'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py',
           'tests/test_industrial_robot_data.py', 'tests/test_robot_coupling_study.py',
           'research/robot-coupling-protocol.md')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise ValueError(message)


class FitFailure(RuntimeError):
    """One predeclared fitting attempt failed; its recipe cannot be selected."""


class WholeStudyTimeout(TimeoutError):
    """Fatal deadline, never converted to a recoverable fit failure."""


def config():
    return {'version': VERSION, 'arms': list(ARMS), 'seeds': list(SEEDS), 'learning_rates': list(RATES),
            'context': 32, 'train_horizon': 64, 'dev_horizon': 128, 'horizons': [64, 128],
            'skip': 64, 'dev_stride': 160, 'direct_fit_stride': 32, 'direct_history': 16,
            'updates': 1024, 'batch_size': 16, 'adam_betas': [.9, .999], 'adam_eps': 1e-8,
            'gradient_clip': 1., 'ridge': 1., 'direct_ridges': [1., 100.],
            'window_seed_offset': 510000, 'fit_cap_seconds': 600., 'wall_cap_seconds': 3600.,
            'simulation_dtype': 'float32', 'ridge_dtype': 'float64', 'normalization_dtype': 'float64',
            'normalization_scope': 'all FIT rows after per-recording skip',
            'partitions': {key: list(PARTITIONS[key]) for key in ('fit', 'dev')},
            'confirmation_access': False, 'official_test_access': False,
            'preprocessing_sha256': PREPROCESSING_SHA256, 'timing_warmups': 3, 'timing_repeats': 20,
            'mean_reduction': .1, 'paired_reduction': .05, 'joint_harm_ratio': 1.1,
            'direct_noninferiority_ratio': 1.05}


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink file required')
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def authenticate(registration):
    path = Path(registration).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    require(plan['version'] == VERSION and plan['config'] == config(), 'exact registered version/config required')
    require(set(plan['sources']) == set(SOURCES), 'exact complete source roster required')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT / name) == expected, 'source pin changed: ' + name)
    require(set(plan['raw_recordings']) == set(PARTITIONS['fit']) | set(PARTITIONS['dev']),
            'raw roster must be exactly FIT7 and DEV2, excluding CONFIRM/TEST')
    for item in plan['raw_recordings'].values():
        require(set(item) == {'path', 'sha256', 'bytes'} and Path(item['path']).is_absolute(), 'raw descriptor schema')
    contract = plan['metadata_contract']
    require(descriptor(contract['path']) == {key: contract[key] for key in ('sha256', 'bytes')}, 'metadata contract pin')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread environment required')
    return plan, hashlib.sha256(raw).hexdigest()


def load_registered(plan, name):
    require(name in plan['raw_recordings'] and name in (*PARTITIONS['fit'], *PARTITIONS['dev']), 'closed partition')
    item = plan['raw_recordings'][name]
    path = Path(item['path'])
    require(path.is_file() and not path.is_symlink(), 'registered regular raw source required')
    raw = path.read_bytes()
    require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'raw MAT pin before decode')
    record = load_recording(raw, name)
    require(record.pins['source_sha256'] == item['sha256'] and record.pins['source_bytes'] == item['bytes'], 'loader source join')
    require(record.pins['preprocessing_sha256'] == PREPROCESSING_SHA256, 'loader preprocessing join')
    return record


def normalizers(records, skip):
    q = np.concatenate([np.asarray(r['q'][skip:], dtype=np.float64) for r in records])
    u = np.concatenate([np.asarray(r['u'][skip:], dtype=np.float64) for r in records])
    require(q.ndim == 2 and q.shape[1] == 6 and q.shape == u.shape and np.isfinite(q).all() and np.isfinite(u).all(), 'finite FIT rows')
    result = {'q_mean': q.mean(0), 'q_std': q.std(0), 'u_mean': u.mean(0), 'u_std': u.std(0)}
    require(all(np.isfinite(value).all() for value in result.values())
            and np.all(result['q_std'] > 0) and np.all(result['u_std'] > 0), 'finite nonconstant FIT channels')
    return result


def normalized_record(record, norm):
    return {'name': record['name'], 'q': (record['q'] - norm['q_mean']) / norm['q_std'],
            'u': (record['u'] - norm['u_mean']) / norm['u_std']}


def quadratic_features(x):
    require(x.shape[-1] == 24, 'quadratic input has24 features')
    if isinstance(x, torch.Tensor):
        i, j = torch.triu_indices(24, 24, device=x.device)
        return torch.cat((torch.ones_like(x[..., :1]), x, x[..., i] * x[..., j]), dim=-1)
    x = np.asarray(x)
    i, j = np.triu_indices(24)
    return np.concatenate((np.ones_like(x[..., :1]), x, x[..., i] * x[..., j]), axis=-1)


def ar2_features(q, prevq, u, prevu):
    return np.concatenate((q, prevq, u, prevu, np.ones((*q.shape[:-1], 1))), axis=-1)


def solve_ridge(phi, targets, penalty=1.):
    require(phi.ndim == 2 and targets.ndim == 2 and len(phi) == len(targets) and len(phi) > 0, 'ridge rows')
    require(np.isfinite(phi).all() and np.isfinite(targets).all() and np.isfinite(penalty) and penalty > 0, 'finite ridge inputs')
    result = np.linalg.solve(phi.T @ phi + penalty * np.eye(phi.shape[1]), phi.T @ targets)
    require(np.isfinite(result).all(), 'finite ridge coefficient')
    return result


def window_batch(records, choices, context, horizon):
    record_ids = np.asarray(choices['record'])
    starts = np.asarray(choices['start'])
    require(record_ids.dtype == starts.dtype == np.dtype('int64') and record_ids.shape == starts.shape
            and record_ids.ndim == 1 and len(starts) > 0, 'int64 window vectors')
    values = {key: [] for key in ('q_context', 'u_context', 'future_u', 'target')}
    for recording, start in zip(record_ids, starts, strict=True):
        require(0 <= recording < len(records), 'valid recording index')
        data = records[int(recording)]
        require(context >= 2 and horizon > 0 and 0 <= start and start + context + horizon <= len(data['q']), 'window bounds')
        values['q_context'].append(data['q'][start:start+context])
        values['u_context'].append(data['u'][start:start+context])
        values['future_u'].append(data['u'][start+context-1:start+context-1+horizon])
        values['target'].append(data['q'][start+context:start+context+horizon])
    return {key: np.stack(value) for key, value in values.items()}


def make_batches(lengths, seed, cfg):
    require(len(lengths) > 0 and all(n >= cfg['skip'] + cfg['context'] + cfg['train_horizon'] for n in lengths), 'FIT windows exist')
    rng = np.random.Generator(np.random.PCG64(seed + cfg['window_seed_offset']))
    shape = (cfg['updates'], cfg['batch_size'])
    record = rng.integers(0, len(lengths), size=shape, dtype=np.int64)
    starts = np.empty(shape, dtype=np.int64)
    for index in np.ndindex(shape):
        high = lengths[int(record[index])] - cfg['context'] - cfg['train_horizon'] + 1
        starts[index] = rng.integers(cfg['skip'], high)
    return {'record': record, 'start': starts}


def dev_windows(length, cfg):
    starts = np.arange(cfg['skip'], length - cfg['context'] - cfg['dev_horizon'] + 1,
                       cfg['dev_stride'], dtype=np.int64)
    require(len(starts) > 0, 'DEV windows exist')
    return starts


def direct_features(batch, history=16):
    q, u, future = batch['q_context'], batch['u_context'], batch['future_u']
    require(q.shape[1] >= history + 1 and future.shape[1] == 128, 'direct history/future geometry')
    # Last past torque ends at C-2. Planned torque starts at C-1 exactly once.
    return np.concatenate((q[:, -history:].reshape(len(q), -1), u[:, -history-1:-1].reshape(len(q), -1),
                           future.reshape(len(q), -1), np.ones((len(q), 1))), axis=1)


def fit_references(records, cfg):
    linear_x, quadratic_x, target = [], [], []
    direct_x, direct_y = [], []
    for index, data in enumerate(records):
        t = np.arange(max(1, cfg['skip']), len(data['q']) - 1)
        features = ar2_features(data['q'][t], data['q'][t-1], data['u'][t], data['u'][t-1])
        linear_x.append(features)
        quadratic_x.append(quadratic_features(features[:, :24]))
        target.append(data['q'][t+1])
        starts = np.arange(cfg['skip'], len(data['q']) - cfg['context'] - cfg['dev_horizon'] + 1,
                           cfg['direct_fit_stride'], dtype=np.int64)
        batch = window_batch(records, {'record': np.full(len(starts), index, dtype=np.int64), 'start': starts},
                             cfg['context'], cfg['dev_horizon'])
        direct_x.append(direct_features(batch, cfg['direct_history']))
        direct_y.append(batch['target'].reshape(len(starts), -1))
    target = np.concatenate(target)
    result = {'linear_frozen': solve_ridge(np.concatenate(linear_x), target, cfg['ridge']).T,
              'quadratic_frozen': solve_ridge(np.concatenate(quadratic_x), target, cfg['ridge']).T}
    for penalty in cfg['direct_ridges']:
        result[f'direct_ridge_{int(penalty)}'] = solve_ridge(np.concatenate(direct_x), np.concatenate(direct_y), penalty)
    return result


class GRUState(NamedTuple):
    q: torch.Tensor
    prevq: torch.Tensor
    prevu: torch.Tensor
    hidden: torch.Tensor


class ARState(NamedTuple):
    q: torch.Tensor
    prevq: torch.Tensor
    prevu: torch.Tensor


class GRUResidual(nn.Module):
    def __init__(self, seed, coefficient):
        super().__init__()
        self.base_weight = nn.Parameter(torch.as_tensor(coefficient, dtype=torch.float32).clone())
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.gru = nn.GRUCell(24, 10, dtype=torch.float32)
            self.head = nn.Linear(10, 6, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def condition(self, q_context, u_context):
        require(q_context.shape == u_context.shape and q_context.ndim == 3 and q_context.shape[1] >= 2
                and q_context.shape[-1] == 6, 'GRU context shape')
        hidden = q_context.new_zeros(len(q_context), 10)
        for t in range(1, q_context.shape[1] - 1):
            x = torch.cat((q_context[:, t], q_context[:, t-1], u_context[:, t], u_context[:, t-1]), dim=-1)
            hidden = self.gru(x, hidden)
        return GRUState(q_context[:, -1].clone(), q_context[:, -2].clone(), u_context[:, -2].clone(), hidden)

    def step(self, inputs, state):
        x = torch.cat((state.q, state.prevq, inputs, state.prevu), dim=-1)
        hidden = self.gru(x, state.hidden)
        prediction = torch.cat((x, torch.ones_like(x[:, :1])), dim=-1) @ self.base_weight.T + self.head(hidden)
        return prediction, GRUState(prediction, state.q, inputs, hidden)

    def forward(self, sequence, state):
        outputs = []
        for t in range(sequence.shape[1]):
            output, state = self.step(sequence[:, t], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class QuadraticAR2(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = nn.Parameter(torch.as_tensor(coefficient, dtype=torch.float32).clone())

    def condition(self, q_context, u_context):
        require(q_context.shape == u_context.shape and q_context.ndim == 3 and q_context.shape[1] >= 2
                and q_context.shape[-1] == 6, 'quadratic context shape')
        return ARState(q_context[:, -1].clone(), q_context[:, -2].clone(), u_context[:, -2].clone())

    def step(self, inputs, state):
        x = torch.cat((state.q, state.prevq, inputs, state.prevu), dim=-1)
        prediction = quadratic_features(x) @ self.coefficient.T
        return prediction, ARState(prediction, state.q, inputs)

    def forward(self, sequence, state):
        outputs = []
        for t in range(sequence.shape[1]):
            output, state = self.step(sequence[:, t], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


def model_for(arm, seed, linear, quadratic):
    if arm in GRAPH_ARMS:
        return JointCoupling(arm, seed, torch.as_tensor(linear, dtype=torch.float32))
    if arm == 'gru_residual':
        return GRUResidual(seed, linear)
    require(arm == 'quadratic_ar2', 'declared neural family')
    return QuadraticAR2(quadratic)


def weights(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def optimizer_arrays(model, optimizer):
    result = {}
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            require(isinstance(value, torch.Tensor), 'tensor Adam state')
            result[name + '/' + key] = value.detach().cpu().numpy().copy()
    return result


def infer(model, batch):
    # Converts only public observed context and planned torques. The target key
    # is deliberately never passed to either conditioning or simulation.
    q = torch.from_numpy(np.asarray(batch['q_context'], dtype=np.float32))
    u = torch.from_numpy(np.asarray(batch['u_context'], dtype=np.float32))
    future = torch.from_numpy(np.asarray(batch['future_u'], dtype=np.float32))
    return model(future, model.condition(q, u))[0]


def train_one(model, records, batches, *, cfg, lr, folder, check=lambda: None, fit_started=None):
    started = time.monotonic() if fit_started is None else fit_started
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder / 'initial.npz', **weights(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    trace, error, status = [], None, 'PASS'
    loop_started = time.monotonic()
    try:
        for update in range(cfg['updates']):
            check()
            if time.monotonic() - started > cfg['fit_cap_seconds']:
                raise FitFailure('single-fit wall cap exceeded')
            choices = {key: value[update] for key, value in batches.items()}
            batch = window_batch(records, choices, cfg['context'], cfg['train_horizon'])
            optimizer.zero_grad(set_to_none=True)
            prediction = infer(model, batch)
            target = torch.from_numpy(np.asarray(batch['target'], dtype=np.float32))
            loss = (prediction - target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FitFailure('nonfinite autoregressive training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=False)
            if not bool(torch.isfinite(norm)):
                raise FitFailure('nonfinite training gradient norm')
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FitFailure('nonfinite updated parameters')
            if not all(np.isfinite(value).all() for value in optimizer_arrays(model, optimizer).values()):
                raise FitFailure('nonfinite updated Adam state')
            trace.append({'update': update + 1, 'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm.detach())})
            check()
            if time.monotonic() - started > cfg['fit_cap_seconds']:
                raise FitFailure('single-fit wall cap exceeded')
    except FitFailure as exc:
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
    except ValueError as exc:
        # Only the qualified graph's exact numerical-output guard is a fit
        # failure. Schema, source and programming errors remain campaign-fatal.
        if str(exc) != 'nonfinite joint-coupling output; no clipping or repair':
            status, error = 'FATAL', {'type': type(exc).__name__, 'message': str(exc)}
            raise
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
    except BaseException as exc:
        status, error = 'FATAL', {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        optimizer_seconds = time.monotonic() - loop_started
        np.savez_compressed(folder / 'final.npz', **weights(model))
        np.savez_compressed(folder / 'optimizer.npz', **optimizer_arrays(model, optimizer))
        write_json(folder / 'trace.json', trace)
        fit_seconds = time.monotonic() - started
        if status == 'PASS' and fit_seconds > cfg['fit_cap_seconds']:
            status, error = 'FAILED', {'type': 'FitFailure', 'message': 'single-fit wall cap exceeded during preservation'}
        receipt = {'status': status, 'error': error, 'completed_updates': len(trace),
                   'requested_updates': cfg['updates'], 'learning_rate': lr, 'optimizer_seconds': optimizer_seconds,
                   'fit_seconds': fit_seconds, 'fit_cap_scope': 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows',
                   'timing_scope': 'optimizer loop including batch construction and finite checks, excluding model construction and saved files',
                   'files': {p.name: descriptor(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        write_json(folder / 'fit-receipt.json', receipt)
    return receipt


def reference_predict(name, coefficients, batch):
    q = np.asarray(batch['q_context'], dtype=np.float64)
    u = np.asarray(batch['u_context'], dtype=np.float64)
    future = np.asarray(batch['future_u'], dtype=np.float64)
    if name == 'persistence':
        return np.repeat(q[:, -1:, :], future.shape[1], axis=1)
    if name == 'velocity':
        times = np.arange(5, dtype=np.float64) - 2
        slope = np.sum(q[:, -5:] * times[None, :, None], axis=1) / np.sum(times**2)
        return q[:, -1:, :] + slope[:, None, :] * np.arange(1, future.shape[1] + 1)[None, :, None]
    if name.startswith('direct_ridge_'):
        return (direct_features(batch) @ coefficients[name]).reshape(len(q), 128, 6)
    require(name in ('linear_frozen', 'quadratic_frozen'), 'known frozen reference')
    current, previous, prevu, outputs = q[:, -1].copy(), q[:, -2].copy(), u[:, -2].copy(), []
    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(future.shape[1]):
            features = ar2_features(current, previous, future[:, t], prevu)
            if name == 'quadratic_frozen':
                features = quadratic_features(features[:, :24])
            prediction = features @ coefficients[name].T
            outputs.append(prediction)
            previous, current, prevu = current, prediction, future[:, t]
    return np.stack(outputs, axis=1)


def metrics(prediction, target, q_std, horizon):
    require(prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[2] == 6
            and 0 < horizon <= prediction.shape[1], 'prediction/target shape')
    require(np.asarray(q_std).shape == (6,) and np.isfinite(q_std).all() and np.all(q_std > 0), 'valid position scales')
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'nonfinite prediction/target')
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.asarray(prediction[:, :horizon], dtype=np.float64) - target[:, :horizon]
        physical = delta * q_std
        standardized_sse = float(np.sum(delta**2))
        physical_mse = np.mean(physical**2, axis=(0, 1))
    require(np.isfinite(standardized_sse) and np.isfinite(physical_mse).all(), 'nonfinite metric arithmetic')
    return {'standardized_rmse': float(np.sqrt(standardized_sse / delta.size)),
            'standardized_sse': standardized_sse, 'scalars': int(delta.size),
            'physical_rmse_deg': float(np.sqrt(np.mean(physical_mse))),
            'per_joint_rmse_deg': np.sqrt(physical_mse).tolist(),
            'windows': len(prediction), 'horizon': horizon}


def valid_metric_row(row):
    if row.get('status') != 'PASS' or not isinstance(row.get('metrics'), dict):
        return False
    metric = row['metrics']
    try:
        scalars = [metric[key] for key in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')]
        joints = metric['per_joint_rmse_deg']
        return (len(joints) == 6 and all(np.isfinite(value) and value >= 0 for value in [*scalars, *joints])
                and type(metric['scalars']) is int and metric['scalars'] > 0
                and type(metric['windows']) is int and metric['windows'] > 0
                and metric['horizon'] == row['horizon'])
    except (KeyError, TypeError, ValueError):
        return False


def scored_rows(common, prediction, target, q_std, cfg, error=None):
    result = []
    for horizon in cfg['horizons']:
        value, local_error = None, error
        if local_error is None:
            try:
                value = metrics(prediction, target, q_std, horizon)
            except ValueError as exc:
                if str(exc) not in ('nonfinite prediction/target', 'nonfinite metric arithmetic'):
                    raise
                local_error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
        result.append({**common, 'horizon': horizon, 'status': 'PASS' if local_error is None else 'FAILED',
                       'error': local_error, 'metrics': value})
    return result


def select_recipes(rows, cfg):
    selected, options = {}, {}
    for arm in cfg['arms']:
        options[arm] = []
        for rate in cfg['learning_rates']:
            subset = [r for r in rows if r['arm'] == arm and r.get('learning_rate') == rate and r['horizon'] == 128]
            keys = {(r['recording'], r['seed']) for r in subset}
            complete = (len(subset) == len(cfg['seeds']) * len(cfg['partitions']['dev'])
                        and keys == {(name, seed) for name in cfg['partitions']['dev'] for seed in cfg['seeds']}
                        and all(valid_metric_row(r) for r in subset))
            score = (float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset)
                                  / sum(r['metrics']['scalars'] for r in subset))) if complete else None)
            options[arm].append({'learning_rate': rate, 'eligible': complete, 'pooled_dev_h128_rmse': score})
        valid = [option for option in options[arm] if option['eligible']]
        selected[arm] = min(valid, key=lambda x: (x['pooled_dev_h128_rmse'], x['learning_rate']))['learning_rate'] if valid else None
    direct = []
    for name in ('direct_ridge_1', 'direct_ridge_100'):
        subset = [r for r in rows if r['arm'] == name and r['horizon'] == 128]
        complete = (len(subset) == len(cfg['partitions']['dev']) and {r['recording'] for r in subset} == set(cfg['partitions']['dev'])
                    and all(valid_metric_row(r) for r in subset))
        score = (float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset) / sum(r['metrics']['scalars'] for r in subset)))
                 if complete else None)
        direct.append({'arm': name, 'eligible': complete, 'pooled_dev_h128_rmse': score})
    valid = [option for option in direct if option['eligible']]
    return {'selected_rates': selected, 'rate_options': options,
            'selected_direct': min(valid, key=lambda x: (x['pooled_dev_h128_rmse'], x['arm']))['arm'] if valid else None,
            'direct_options': direct, 'selection_scope': 'pooled internal DEV H128, no CONFIRM or official TEST access'}


def evaluate_rule(rows, selection, cfg):
    conditions = []
    chosen = selection['selected_rates']
    finite = all(chosen.get(arm) is not None for arm in cfg['arms']) and selection['selected_direct'] is not None
    fixed = ('linear_frozen', 'quadratic_frozen', 'persistence', 'velocity')
    fixed_rows = [r for r in rows if r['arm'] in fixed and r['horizon'] == 128]
    finite = (finite and len(fixed_rows) == len(fixed) * len(cfg['partitions']['dev'])
              and {(r['arm'], r['recording']) for r in fixed_rows} == {(arm, name) for arm in fixed for name in cfg['partitions']['dev']}
              and all(valid_metric_row(r) for r in fixed_rows))
    conditions.append({'name': 'all_mandatory_selected_recipes_finite', 'passed': bool(finite)})
    detail = {}
    for name in cfg['partitions']['dev']:
        current = [r for r in rows if r['recording'] == name and r['horizon'] == 128 and valid_metric_row(r)]
        neural = {arm: {r['seed']: r['metrics'] for r in current if r['arm'] == arm and r.get('learning_rate') == chosen.get(arm)}
                  for arm in cfg['arms']}
        ref = {r['arm']: r['metrics'] for r in current if r['arm'] in REFERENCES}
        means = {arm: float(np.mean([v['standardized_rmse'] for v in values.values()]))
                 for arm, values in neural.items() if set(values) == set(cfg['seeds'])}
        candidate = means.get('chain_memory')
        detail[name] = {'selected_family_mean_standardized_rmse': means,
                        'reference_standardized_rmse': {key: value['standardized_rmse'] for key, value in ref.items()}}
        for control in ('chain_instant', 'rewired_memory', 'gru_residual', 'quadratic_ar2', *fixed):
            value = means.get(control) if control in cfg['arms'] else ref.get(control, {}).get('standardized_rmse')
            conditions.append({'name': name + '/mean_10pct/' + control,
                               'passed': candidate is not None and value is not None and candidate <= (1 - cfg['mean_reduction']) * value})
        for control in ('chain_instant', 'rewired_memory', 'gru_residual', 'quadratic_ar2'):
            for seed in cfg['seeds']:
                left, right = neural['chain_memory'].get(seed), neural[control].get(seed)
                conditions.append({'name': f'{name}/seed{seed}_5pct/{control}',
                                   'passed': left is not None and right is not None
                                   and left['standardized_rmse'] <= (1 - cfg['paired_reduction']) * right['standardized_rmse']})
        for joint in range(6):
            available = set(neural['chain_memory']) == set(neural['gru_residual']) == set(cfg['seeds'])
            left = np.mean([r['per_joint_rmse_deg'][joint] for r in neural['chain_memory'].values()]) if available else None
            right = np.mean([r['per_joint_rmse_deg'][joint] for r in neural['gru_residual'].values()]) if available else None
            conditions.append({'name': f'{name}/joint{joint}_no_10pct_harm_vs_gru',
                               'passed': bool(available and left <= cfg['joint_harm_ratio'] * right)})
        direct = ref.get(selection['selected_direct'], {}).get('standardized_rmse')
        conditions.append({'name': name + '/within_5pct_direct_history',
                           'passed': candidate is not None and direct is not None and candidate <= cfg['direct_noninferiority_ratio'] * direct})
    passed = sum(bool(item['passed']) for item in conditions)
    return {'status': 'ADVANCE_TO_SEPARATELY_AUTHORIZED_CONFIRMATION' if passed == len(conditions) else 'DO_NOT_ADVANCE_COUPLING',
            'passed': passed, 'total': len(conditions), 'conditions': conditions, 'details': detail,
            'scope': 'internal DEV screening after two-rate selection, not held-out confirmation or official benchmark performance'}


def state_count(arm):
    return 28 if arm in ('chain_memory', 'rewired_memory', 'gru_residual') else 18


def resource_model(model, arm):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'state_scalars': state_count(arm), 'state_bytes': 4 * state_count(arm),
            'buffer_bytes': sum(p.numel() * p.element_size() for p in model.buffers()),
            'normalizer_bytes': 24 * 8, 'dtype': 'float32',
            'request_input_bytes_float64': 6 * (32 + 32 + 128) * 8, 'request_output_bytes_float64': 128 * 6 * 8,
            'temporary_workspace': 'not measured',
            'scope': 'one stream; weights, explicit recurrent state, declared buffers and normalization; workspace excluded'}


def timed_request(arm, model, coefficients, physical_batch, norm, cfg, check):
    def request():
        check()
        batch = {'q_context': (physical_batch['q_context'] - norm['q_mean']) / norm['q_std'],
                 'u_context': (physical_batch['u_context'] - norm['u_mean']) / norm['u_std'],
                 'future_u': (physical_batch['future_u'] - norm['u_mean']) / norm['u_std']}
        with torch.no_grad():
            prediction = infer(model, batch).numpy().astype(np.float64) if model is not None else reference_predict(arm, coefficients, batch)
        output = prediction * norm['q_std'] + norm['q_mean']
        require(np.isfinite(output).all(), 'finite timed full forecast')
        return output
    for _ in range(cfg['timing_warmups']):
        request()
    durations = []
    for _ in range(cfg['timing_repeats']):
        started = time.perf_counter()
        request()
        durations.append(time.perf_counter() - started)
    return {'seconds': durations, 'median_seconds': float(np.median(durations)), 'p95_seconds': float(np.percentile(durations, 95)),
            'scope': 'batch1 physical-array normalization/cast,32-point conditioning,128-step forecast and denormalization; no target input'}


def run(registration, output):
    plan, registration_sha = authenticate(registration)
    cfg = config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    models, fits, rows, resources = {}, [], [], []
    def check():
        if time.monotonic() - started > cfg['wall_cap_seconds']:
            raise WholeStudyTimeout('whole study wall cap exceeded')
    try:
        snapshot = output / 'sources'
        snapshot.mkdir()
        for name in SOURCES:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        write_json(output / 'registration.json', plan)
        write_json(output / 'admission.json', {'registration_path': str(Path(registration).resolve()), 'registration_sha256': registration_sha,
                                              'sources': plan['sources'], 'confirmation_access': False, 'official_test_access': False})
        fit_raw = []
        for name in PARTITIONS['fit']:
            check()
            record = load_registered(plan, name)
            fit_raw.append({'name': name, 'q': record.q, 'u': record.torque})
            np.savez_compressed(output / ('fit-data-' + name + '.npz'), q=record.q, u=record.torque, raw_indices=record.raw_indices)
            check()
        norm = normalizers(fit_raw, cfg['skip'])
        np.savez_compressed(output / 'normalizers.npz', **norm)
        fit_data = [normalized_record(record, norm) for record in fit_raw]
        reference_started = time.monotonic()
        coefficients = fit_references(fit_data, cfg)
        check()
        np.savez_compressed(output / 'references.npz', **coefficients)
        write_json(output / 'reference-fit.json', {'seconds': time.monotonic() - reference_started,
            'scope': 'all float64 FIT-only ridge designs and solves, excludes source decode and normalization',
            'fit_recordings': list(PARTITIONS['fit']), 'ridge_penalty_includes_intercept': True})
        for seed in SEEDS:
            batches = make_batches([len(data['q']) for data in fit_data], seed, cfg)
            np.savez_compressed(output / f'batches-{seed}.npz', **batches)
            for rate_index, rate in enumerate(RATES):
                for arm in ARMS:
                    check()
                    key = f'{arm}-{seed}-lr{rate_index}'
                    construct_started = time.monotonic()
                    model = model_for(arm, seed, coefficients['linear_frozen'], coefficients['quadratic_frozen'])
                    construction_seconds = time.monotonic() - construct_started
                    fit = train_one(model, fit_data, batches, cfg=cfg, lr=rate, folder=output / key, check=check, fit_started=construct_started)
                    fits.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate, 'construction_seconds': construction_seconds,
                                 'fit': fit, 'resources': resource_model(model, arm), 'batches': descriptor(output / f'batches-{seed}.npz')})
                    models[key] = model
                    write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
        require(len(fits) == 30, 'all30 fit attempts before DEV decode')
        write_json(output / 'checkpoint-barrier.json', {'fit_attempts': len(fits), 'dev_decodes': 0, 'confirmation_decodes': 0,
            'final_checkpoints': {fit['key']: descriptor(output / fit['key'] / 'final.npz') for fit in fits},
            'fit_statuses': {fit['key']: fit['fit']['status'] for fit in fits}})
        dev_data, physical_for_timing = [], None
        for name in PARTITIONS['dev']:
            check()
            record = load_registered(plan, name)
            physical = {'name': name, 'q': record.q, 'u': record.torque}
            np.savez_compressed(output / ('dev-data-' + name + '.npz'), q=record.q, u=record.torque, raw_indices=record.raw_indices)
            data = normalized_record(physical, norm)
            starts = dev_windows(len(data['q']), cfg)
            choices = {'record': np.zeros(len(starts), dtype=np.int64), 'start': starts}
            batch = window_batch([data], choices, cfg['context'], cfg['dev_horizon'])
            dev_data.append((name, batch))
            np.savez_compressed(output / ('dev-windows-' + name + '.npz'), starts=starts, target=batch['target'])
            if physical_for_timing is None:
                physical_for_timing = window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, cfg['context'], cfg['dev_horizon'])
            for fit in fits:
                common = {'recording': name, 'arm': fit['arm'], 'seed': fit['seed'], 'learning_rate': fit['learning_rate'], 'fit_key': fit['key']}
                error, prediction = None, None
                if fit['fit']['status'] == 'PASS':
                    try:
                        with torch.no_grad():
                            prediction = infer(models[fit['key']], batch).numpy().astype(np.float64)
                        require(np.isfinite(prediction).all(), 'nonfinite DEV autoregression')
                    except ValueError as exc:
                        if str(exc) not in ('nonfinite DEV autoregression', 'nonfinite joint-coupling output; no clipping or repair'):
                            raise
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                else:
                    error = {'type': 'FailedTrainingAttempt', 'message': 'not a completed trained model'}
                if prediction is not None:
                    np.savez_compressed(output / ('prediction-' + name + '-' + fit['key'] + '.npz'), prediction=prediction)
                rows.extend(scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error))
                check()
            for arm in REFERENCES:
                prediction = reference_predict(arm, coefficients, batch)
                finite = bool(np.isfinite(prediction).all())
                np.savez_compressed(output / ('prediction-' + name + '-' + arm + '.npz'), prediction=prediction)
                rows.extend(scored_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None},
                            prediction, batch['target'], norm['q_std'], cfg, None if finite else {'type': 'NonfiniteReference'}))
                check()
        selection = select_recipes(rows, cfg)
        result = evaluate_rule(rows, selection, cfg)
        for fit in fits:
            if selection['selected_rates'][fit['arm']] == fit['learning_rate']:
                timing = timed_request(fit['arm'], models[fit['key']], coefficients, physical_for_timing, norm, cfg, check)
                resources.append({'arm': fit['arm'], 'seed': fit['seed'], 'learning_rate': fit['learning_rate'],
                                  **fit['resources'], 'timing': timing, 'optimizer_seconds': fit['fit']['optimizer_seconds']})
        for arm in (*REFERENCES[:4], selection['selected_direct']):
            if arm is None:
                continue
            valid = all(row['status'] == 'PASS' for row in rows if row['arm'] == arm)
            parameter_count = int(coefficients[arm].size) if arm in coefficients else 0
            resources.append({'arm': arm, 'seed': None, 'learning_rate': None, 'parameters': parameter_count,
                'parameter_bytes': parameter_count * 8, 'normalizer_bytes': 192, 'buffer_bytes': 0,
                'state_scalars': 18 if arm in REFERENCES[:2] else (30 if arm == 'velocity' else (6 if arm == 'persistence' else 192)),
                'state_bytes': 8 * (18 if arm in REFERENCES[:2] else (30 if arm == 'velocity' else (6 if arm == 'persistence' else 192))),
                'request_input_bytes_float64': 6 * (32 + 32 + 128) * 8, 'request_output_bytes_float64': 128 * 6 * 8,
                'temporary_workspace': 'not measured', 'dtype': 'float64', 'scope': 'stored coefficients, required history and normalization; future inputs and workspace excluded',
                'timing': timed_request(arm, None, coefficients, physical_for_timing, norm, cfg, check) if valid else None})
        write_json(output / 'results.json', {'version': VERSION, 'config': cfg, 'rows': rows, 'selection': selection, 'result': result})
        write_json(output / 'resources.json', resources)
        write_json(output / 'fits.json', fits)
        check()
        after, after_sha = authenticate(registration)
        require(after == plan and after_sha == registration_sha, 'original registration unchanged')
        write_json(output / 'manifest.json', {'files': {str(path.relative_to(output)): descriptor(path)
                   for path in sorted(output.rglob('*')) if path.is_file()}})
        check()
        write_json(output / 'receipt.json', {'status': 'PASS', 'registration_sha256': registration_sha,
            'elapsed_seconds': time.monotonic() - started, 'fits': len(fits), 'rows': len(rows), 'sources_before': plan['sources'],
            'sources_after': after['sources'], 'fit_decodes': 7, 'dev_decodes': 2, 'confirmation_decodes': 0, 'official_test_decodes': 0,
            'scientific_result': result['status']})
    except BaseException as exc:
        write_json(output / 'failure.json', {'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc),
            'elapsed_seconds': time.monotonic() - started, 'completed_fit_attempts': len(fits), 'completed_metric_rows': len(rows),
            'registration_sha256': registration_sha, 'confirmation_decodes': 0, 'official_test_decodes': 0})
        raise
    return {'result': result, 'fits': len(fits), 'rows': len(rows), 'selection': selection}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registration, args.output)


if __name__ == '__main__':
    main()
