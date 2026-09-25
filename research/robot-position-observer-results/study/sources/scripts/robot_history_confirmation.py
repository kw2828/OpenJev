"""Fixed-checkpoint internal confirmation; no fitting or recipe selection.

Only two explicitly admitted raw confirmation recordings may be decoded. The
official TEST has no runner interface. Numeric failures remain failed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import robot_history_initialization_study as history
import scipy
import torch

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.industrial_robot_data import PREPROCESSING_SHA256, load_recording
from openjev.research.suspend_clock import SuspendClock

old = history.old
ROOT = history.ROOT
VERSION = 'robot-history-confirmation-v1'
PRIMARY, ALL_ARMS, REFERENCES = history.ARMS, history.ALL_ARMS, history.REFERENCES
SEEDS = (8101, 8102, 8103)
FIXED_RATES = {'last_two': .001, 'local_affine': .003, 'temporal_affine': .003,
               'dense_bounded': .001, 'dense_unbounded': .003, 'gru32': .003,
               'legacy_instant': .001, 'gru10': .003}
CONFIRM = ('recording_2021_12_15_22H_41M.mat', 'recording_2021_12_15_22H_50M.mat')
PARENT = ROOT / 'output/robot-history-initialization-study-v1'
PARENT_ENGINEERING = ROOT / 'output/robot-history-initialization-engineering-v1'
PARENT_AUDIT = ROOT / 'output/robot-history-initialization-audit-v1/audit.json'
PARENT_SHA = '628b038c910bf37246e939b104d70530f6783a7fbc67f79afeb79b8813738994'
ARCHIVE_SHA = '9011509cf901dcb50a8de1e4a5a0fcf6cb8bed2f64945ba0fbbfb8e0b54886fe'
COMMON_PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                   'causal_ridge_100.npz', 'causal_ridge_100.json')


def model_key(arm, seed):
    old.require(arm in ALL_ARMS and seed in SEEDS, 'fixed model identity')
    return f'{arm}-{seed}-lr{0 if FIXED_RATES[arm] == .001 else 1}'


PAYLOADS = (*COMMON_PAYLOADS, *(model_key(arm, seed) + '/final.npz' for arm in ALL_ARMS for seed in SEEDS))
SOURCES = (*history.SOURCES, 'scripts/plot_robot_history_initialization.py',
           'scripts/audit_robot_history_initialization.py', 'scripts/robot_history_confirmation.py',
           'tests/test_robot_history_confirmation.py', 'research/robot-history-confirmation-protocol.md',
           'scripts/audit_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py')
QUALIFICATION_SOURCES = ('scripts/robot_history_confirmation.py', 'scripts/audit_robot_history_confirmation.py',
                         'scripts/launch_robot_history_confirmation.py', 'tests/test_robot_history_confirmation.py',
                         'tests/test_audit_robot_history_confirmation.py')
NUMERIC_ERRORS = ('nonfinite LPV output; no rollout clipping or repair',
                  'nonfinite structured transition output; no repair',
                  'finite CPU tensor with exact shape/dtype: initializer features')


def config():
    return {'version': VERSION, 'arms': list(ALL_ARMS), 'primary_arms': list(PRIMARY), 'references': list(REFERENCES),
            'seeds': list(SEEDS), 'fixed_rates': dict(FIXED_RATES), 'partitions': {'confirm': list(CONFIRM)},
            'context': 32, 'horizon': 128, 'horizons': [64, 128], 'skip': 64, 'window_stride': 160,
            'samples_per_recording': 3636, 'windows_per_recording': 22, 'timing_warmups': 3, 'timing_repeats': 20,
            'wall_cap_seconds': 900., 'mean_reduction': .05, 'file_harm_ratio': 1.02, 'latency_ratio': 1.25,
            'preprocessing_sha256': PREPROCESSING_SHA256, 'confirmation_access': True, 'official_test_access': False,
            'fits': 0, 'normalizer_refits': 0, 'reference_refits': 0, 'recipe_selection_calls': 0}


def _pin(item):
    old.require(isinstance(item, dict) and set(item) == {'path', 'sha256', 'bytes'}
                and Path(item['path']).is_absolute(), 'absolute descriptor required')
    value = old.descriptor(item['path'])
    old.require(value == {k: item[k] for k in ('sha256', 'bytes')}, 'registered bytes changed: ' + item['path'])
    return value


def authenticate(registration):
    """Complete original-source and parent admission before any numerical decode."""
    raw = Path(registration).read_bytes()
    plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact fixed confirmation config')
    old.require(set(plan['sources']) == set(SOURCES) and len(SOURCES) == 36, 'all36 source files')
    for name, expected in plan['sources'].items():
        old.require(old.descriptor(ROOT / name) == expected, 'source drift: ' + name)
    old.require(all(os.environ.get(k) == '1' for k in old.THREADS), 'single-thread environment')
    from plot_robot_history_initialization import authenticate as admit_parent
    parent = admit_parent(PARENT, PARENT_AUDIT, PARENT_ENGINEERING)
    old.require(plan['parent_registration_sha256'] == PARENT_SHA
                and all(parent['plan']['sources'][n] == plan['sources'][n] for n in history.SOURCES), 'unchanged inherited sources')
    selection = parent['audit']['results']['selection']
    old.require(selection['selected_rates'] == FIXED_RATES and parent['audit']['results']['result']['passed'] == 5
                and parent['audit']['results']['result']['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL', 'fixed audited parent choices')
    paths = {'manifest': PARENT / 'manifest.json', 'receipt': PARENT / 'receipt.json',
             'process': PARENT_ENGINEERING / 'run-process-01.json', 'audit': PARENT_AUDIT,
             'audit_process': PARENT_ENGINEERING / 'audit-process-01.json'}
    old.require(set(plan['parent_closure']) == set(paths), 'complete original parent closure')
    for name, path in paths.items():
        item = plan['parent_closure'][name]
        old.require(Path(item['path']) == path and _pin(item) == parent['inputs'][str(path)], 'parent closure join: ' + name)
    inventory = json.loads((PARENT / 'manifest.json').read_text())['files']
    old.require(set(plan['parent_payloads']) == set(PAYLOADS), 'exact24 finals and6 inherited reference payloads')
    for name, item in plan['parent_payloads'].items():
        old.require(Path(item['path']) == PARENT / name and _pin(item) == inventory[name], 'selected parent payload: ' + name)
    launcher = ROOT / 'scripts/launch_robot_history_confirmation.py'
    old.require(old.descriptor(launcher) == plan['launcher'], 'launcher source drift')
    _pin(plan['qualification'])
    qualification = json.loads(Path(plan['qualification']['path']).read_text())
    old.require(qualification['status'] == 'PASS' and qualification['sources_unchanged'] is True
                and qualification['sources'] == plan['sources'] and qualification['launcher'] == plan['launcher']
                and qualification['thread_env'] == dict.fromkeys(old.THREADS, '1'), 'qualified exact source closure')
    commands = [['.venv/bin/ruff', 'check', *QUALIFICATION_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                 'tests/test_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py']]
    old.require(len(qualification['commands']) == 2, 'both original qualification commands')
    for row, command in zip(qualification['commands'], commands, strict=True):
        old.require(row['command'] == command and row['returncode'] == 0
                    and old.descriptor(row['log'])['sha256'] == row['sha256'], 'qualification argv/log join')
    old.require(set(plan['raw_recordings']) == set(CONFIRM), 'exact2 CONFIRM records only; TEST unsupported')
    inputs = ROOT / 'output/robot-history-confirmation-inputs-v1'
    archive = ROOT / 'output/robot-data-engineering-v1/raw-bundle-attempt-01.rar'
    extraction_path = ROOT / 'output/robot-history-confirmation-engineering-v1/raw-extraction-01.json'
    old.require(Path(plan['archive']['path']) == archive and plan['archive']['sha256'] == ARCHIVE_SHA
                and Path(plan['extraction']['path']) == extraction_path, 'original archive/extraction identity')
    _pin(plan['extraction'])
    extraction = json.loads(extraction_path.read_text())
    command = ['/usr/bin/bsdtar', '-xf', str(archive), '-C', str(inputs), *['raw_data/' + n for n in CONFIRM]]
    old.require(extraction['scope'] == 'Opaque extraction and hashing only; no numeric decoding'
                and extraction['command'] == command and extraction['returncode'] == 0
                and type(extraction['elapsed_seconds']) in (float, int) and math.isfinite(extraction['elapsed_seconds'])
                and extraction['elapsed_seconds'] > 0 and extraction['archive'] == plan['archive']
                and extraction['raw_recordings'] == plan['raw_recordings'], 'opaque extraction provenance')
    _pin(plan['archive'])
    for name, item in plan['raw_recordings'].items():
        old.require(Path(item['path']) == inputs / 'raw_data' / name, 'registered confirmation source path')
        _pin(item)
    return plan, hashlib.sha256(raw).hexdigest()


def load_confirmation(plan, name):
    old.require(name in CONFIRM and set(plan['raw_recordings']) == set(CONFIRM), 'only registered confirmation allowed')
    item = plan['raw_recordings'][name]
    _pin(item)
    record = load_recording(Path(item['path']), name, allow_confirmation=True)
    old.require(record.name == name and record.partition == 'confirm'
                and record.pins['source_sha256'] == item['sha256'] and record.pins['source_bytes'] == item['bytes']
                and record.pins['preprocessing_sha256'] == PREPROCESSING_SHA256, 'raw loader identity/preprocessing join')
    return {'name': name, 'q': record.q, 'u': record.torque, 'raw_indices': record.raw_indices}


def make_windows(record, norm, cfg):
    old.require(record['name'] in CONFIRM and all(record[k].shape == (3636, 6) and record[k].dtype == np.float64
                and np.isfinite(record[k]).all() for k in ('q', 'u')), 'finite complete confirmation recording')
    old.require(record['raw_indices'].dtype == np.int64
                and np.array_equal(record['raw_indices'], np.arange(0, 90881, 25, dtype=np.int64)), 'causal decimation indices')
    old.require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'} and all(x.dtype == np.float64 and x.shape == (6,)
                and np.isfinite(x).all() for x in norm.values()) and np.all(norm['q_std'] > 0) and np.all(norm['u_std'] > 0), 'inherited normalizer schema')
    starts = np.arange(cfg['skip'], len(record['q'])-cfg['context']-cfg['horizon']+1, cfg['window_stride'], dtype=np.int64)
    old.require(np.array_equal(starts, 64 + 160*np.arange(22, dtype=np.int64)), 'fixed22 within-record windows')
    choices = {'record': np.zeros(22, dtype=np.int64), 'start': starts}
    physical = old.window_batch([record], choices, 32, 128)
    normalized = {'name': record['name'], 'q': (record['q']-norm['q_mean'])/norm['q_std'],
                  'u': (record['u']-norm['u_mean'])/norm['u_std']}
    return old.window_batch([normalized], choices, 32, 128), physical, starts


def fixed_rule(rows, resources, cfg):
    """Five original margins, with fixed recipes and no confirmation-based selection."""
    old.require(cfg == config(), 'fixed confirmation rule config')
    recordings = cfg['partitions']['confirm']
    key = lambda r: (r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon'])
    expected = {(n, a, s, FIXED_RATES[a], h) for n in recordings for a in ALL_ARMS for s in SEEDS for h in (64, 128)}
    expected |= {(n, a, None, None, h) for n in recordings for a in REFERENCES for h in (64, 128)}
    old.require(len(rows) == 112 and len({key(r) for r in rows}) == 112 and {key(r) for r in rows} == expected, 'all112 fixed identity rows')
    details, means, costs = {}, {}, {}
    for name in recordings:
        per_arm, paired = {}, {}
        for arm in (*ALL_ARMS, *REFERENCES):
            subset = [r for r in rows if r['recording'] == name and r['arm'] == arm and r['horizon'] == 128]
            if all(old.valid_metric_row(r) for r in subset):
                per_arm[arm] = float(np.mean([r['metrics']['standardized_rmse'] for r in subset]))
                paired[arm] = {r['seed']: r['metrics']['standardized_rmse'] for r in subset}
        differences = {a: {str(s): paired['temporal_affine'][s]-paired[a][s] for s in SEEDS}
                       for a in PRIMARY[:2] if a in paired and 'temporal_affine' in paired}
        details[name] = {'means': per_arm, 'paired_primary_differences': differences}
    expected_costs = {(a, s) for a in ALL_ARMS for s in SEEDS} | {(a, None) for a in REFERENCES}
    actual_costs = [(r['arm'], r['seed']) for r in resources]
    old.require(len(actual_costs) == len(set(actual_costs)) == 28 and set(actual_costs) == expected_costs,
                'all28 declared cost attempts, including failures')
    for arm in (*ALL_ARMS, *REFERENCES):
        if all(arm in details[n]['means'] for n in recordings):
            means[arm] = float(np.mean([details[n]['means'][arm] for n in recordings]))
        costs[arm] = None
        subset = [r for r in resources if r['arm'] == arm]
        seeds = set(SEEDS) if arm in ALL_ARMS else {None}
        if len(subset) != len(seeds) or {r['seed'] for r in subset} != seeds:
            continue
        if any(r.get('status') != 'PASS' or r.get('error') is not None or r.get('timing') is None for r in subset):
            continue
        if arm in ALL_ARMS and any(r['learning_rate'] != FIXED_RATES[arm] for r in subset):
            continue
        times = [r['timing']['median_seconds'] for r in subset]
        sizes = [[r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')] for r in subset]
        if (all(type(t) in (float, int) and math.isfinite(t) and t > 0 for t in times)
                and all(type(v) is int and v >= 0 for parts in sizes for v in parts)):
            costs[arm] = {'latency': float(np.median(times)), 'bytes': max(map(sum, sizes))}
    value = means.get('temporal_affine')
    complete = all(a in means for a in PRIMARY)
    improvement = value is not None and all(means.get(a, 0) > 0 and value <= (1-cfg['mean_reduction'])*means[a] for a in PRIMARY[:2])
    no_harm = all(all(a in d['means'] for a in PRIMARY) and d['means']['temporal_affine'] <= cfg['file_harm_ratio']
                  * min(d['means'][a] for a in PRIMARY[:2]) for d in details.values())
    left, baseline = costs['temporal_affine'], costs['last_two']
    latency = left is not None and baseline is not None and left['latency'] <= cfg['latency_ratio']*baseline['latency']
    frontier = all(a in means and costs[a] is not None for a in (*ALL_ARMS, *REFERENCES))
    dominators = []
    if frontier:
        point = (value, left['latency'], left['bytes'])
        for arm in (*ALL_ARMS, *REFERENCES):
            if arm == 'temporal_affine':
                continue
            other = (means[arm], costs[arm]['latency'], costs[arm]['bytes'])
            if all(a <= b for a, b in zip(other, point, strict=True)) and any(a < b for a, b in zip(other, point, strict=True)):
                dominators.append(arm)
    conditions = [{'name': name, 'passed': bool(flag)} for name, flag in (
        ('primary_recipes_complete', complete), ('equal_file_mean_5pct_vs_both_locals', improvement),
        ('each_file_within_2pct_best_local', no_harm), ('latency_within_125pct_last_two', latency),
        ('complete_frontier_not_dominated', frontier and not dominators))]
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'CONFIRMED_HISTORY_INITIALIZATION' if passed == 5 else 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION',
            'passed': passed, 'total': 5, 'conditions': conditions, 'fixed_rates': dict(FIXED_RATES), 'details': details,
            'equal_file_means': means, 'costs': costs, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'fixed-checkpoint internal recording confirmation; not official TEST, independent robots, novelty or control'}


def prediction_attempt(predict, batch):
    """Only established inference numeric failures become retained failed attempts."""
    try:
        with torch.no_grad():
            prediction = predict({k: batch[k] for k in ('q_context', 'u_context', 'future_u')})
    except old.FitFailure as exc:
        if str(exc) not in NUMERIC_ERRORS:
            raise
        return None, {'type': 'NonfiniteEvaluation', 'message': str(exc)}
    except ValueError as exc:
        if str(exc) != 'nonfinite ridge prediction':
            raise
        return None, {'type': 'NonfiniteEvaluation', 'message': str(exc)}
    old.require(isinstance(prediction, np.ndarray) and prediction.dtype == np.float64
                and prediction.shape == (22, 128, 6), 'full fixed forecast schema')
    return prediction, None


def timing_attempt(predict, physical, norm, cfg):
    """Every scheduled instance is timed; only a numerical failure is retained."""
    try:
        timing = history.timed_request(predict, physical, norm, cfg)
        return {'status': 'PASS', 'error': None, 'timing': timing}
    except old.FitFailure as exc:
        if str(exc) not in NUMERIC_ERRORS:
            raise
        return {'status': 'FAILED', 'error': {'type': 'NonfiniteTiming', 'message': str(exc)}, 'timing': None}
    except ValueError as exc:
        if str(exc) not in ('finite complete timed request', 'nonfinite ridge prediction'):
            raise
        return {'status': 'FAILED', 'error': {'type': 'NonfiniteTiming', 'message': str(exc)}, 'timing': None}


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode()+b'\0'+str(value.dtype).encode()+repr(value.shape).encode()+value.tobytes())
    return digest.hexdigest()


def load_parent_arrays(plan, name):
    old.require(name in PAYLOADS and name.endswith('.npz'), 'declared inherited numerical payload')
    _pin(plan['parent_payloads'][name])
    with np.load(plan['parent_payloads'][name]['path'], allow_pickle=False) as data:
        return {k: data[k].copy(order='K') for k in data.files}


def run(registration, output):
    clock = SuspendClock()
    started_ns, started = clock.now_ns(), time.monotonic()
    plan, sha = authenticate(registration)
    cfg = config()
    output = Path(output)
    old.require(output.resolve() == ROOT / 'output/robot-history-confirmation-v1', 'registered confirmation output')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    models, model_records, rows, attempts, resources = {}, [], [], [], []
    def check():
        if clock.now_ns()-started_ns >= int(cfg['wall_cap_seconds']*1e9):
            raise old.WholeStudyTimeout('native whole-confirmation deadline exceeded')
    try:
        check()
        old.write_json(output / 'runtime.json', {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                       'torch': torch.__version__, 'platform': platform.platform(), 'machine': platform.machine(),
                       'thread_env': {k: os.environ.get(k) for k in old.THREADS}, 'torch_threads': torch.get_num_threads(),
                       'clock': clock.backend, 'timing_scope': 'complete physical request plus uniform native deadline callback'})
        with (output / 'registration.json').open('xb') as handle:
            handle.write(Path(registration).read_bytes())
        for name in SOURCES:
            path = output / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            old.require(old.descriptor(path) == plan['sources'][name], 'exact source snapshot')
        for name, item in plan['parent_payloads'].items():
            check(); _pin(item)
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(Path(item['path']).read_bytes())
            old.require(old.descriptor(path) == {k: item[k] for k in ('sha256', 'bytes')}, 'exact inherited copy')
        norm = load_parent_arrays(plan, 'normalizers.npz')
        old.require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'} and all(v.dtype == np.float64 and v.shape == (6,)
                    and np.isfinite(v).all() for v in norm.values()) and np.all(norm['q_std'] > 0) and np.all(norm['u_std'] > 0), 'inherited FIT normalization')
        linear_data = load_parent_arrays(plan, 'linear.npz')
        old.require(set(linear_data) == {'coefficient'}, 'linear field')
        linear = linear_data['coefficient']
        old.require(linear.dtype == np.float64 and linear.shape == (6, 25) and np.isfinite(linear).all(), 'inherited linear schema')
        ridges = {}
        for arm in REFERENCES[:2]:
            arrays = load_parent_arrays(plan, arm+'.npz')
            old.require(set(arrays) == {f'h{h:03d}' for h in range(1, 129)}, 'all128 causal horizon coefficients')
            metadata = json.loads((output / (arm+'.json')).read_text())
            ridges[arm] = CausalRobotRidge(tuple(arrays[f'h{h:03d}'] for h in range(1, 129)), metadata['penalty'], metadata['fit_rows'])
            old.require(ridges[arm].metadata() == {k: v for k, v in metadata.items() if k != 'fit_seconds'}, 'unchanged causal reference metadata')
        ledger_path = PARENT / 'fits.json'
        ledger_pin = old.descriptor(ledger_path)
        old.require(ledger_pin == json.loads((PARENT/'manifest.json').read_text())['files']['fits.json'], 'historical fit ledger pin')
        ledger = json.loads(ledger_path.read_text())
        inherited = {r['key']: r for r in ledger}
        old.require(len(ledger) == len(inherited) == 48, 'complete original historical fit ledger')
        for arm in ALL_ARMS:
            for seed in SEEDS:
                check()
                key = model_key(arm, seed)
                arrays = load_parent_arrays(plan, key+'/final.npz')
                model = history.model_for(arm, seed, linear)
                template = model.state_dict()
                old.require(set(arrays) == set(template) and all(arrays[k].dtype == np.float32
                            and arrays[k].shape == tuple(v.shape) and np.isfinite(arrays[k]).all() for k, v in template.items()), 'qualified checkpoint shape/dtype')
                model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                old.require(all(p.grad is None for p in model.parameters()), 'zero inference gradient storage')
                models[key] = model
                parent_fit = inherited[key]
                old.require(parent_fit['arm'] == arm and parent_fit['seed'] == seed and parent_fit['learning_rate'] == FIXED_RATES[arm]
                            and parent_fit['effective_status'] == 'PASS' and parent_fit['fit']['status'] == 'PASS'
                            and parent_fit['fit']['files']['final.npz'] == {k: plan['parent_payloads'][key+'/final.npz'][k]
                                                                         for k in ('sha256', 'bytes')}, 'selected historical recipe/checkpoint join')
                model_records.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': FIXED_RATES[arm],
                                      'checkpoint': plan['parent_payloads'][key+'/final.npz'], 'state_sha256': parameter_hash(model),
                                      'resources': history.resource_model(model, arm), 'historical_fit': parent_fit['fit'],
                                      'parent_fit_ledger': {'path': str(ledger_path), **ledger_pin},
                                      'training_scope': 'historical parent fit only; zero new updates'})
        old.write_json(output / 'models.json', model_records)
        old.write_json(output / 'checkpoint-barrier.json', {'checkpoint_count': 24, 'raw_decodes': 0, 'fits': 0,
                       'fixed_rates': dict(FIXED_RATES), 'checkpoints': {r['key']: old.descriptor(output/r['key']/'final.npz') for r in model_records},
                       'state_sha256': {r['key']: r['state_sha256'] for r in model_records}})
        physical_first = None
        def predictor(arm, key=None):
            def predict(batch):
                check()
                if key is not None:
                    return old.infer(models[key], batch).numpy().astype(np.float64)
                return predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges else old.reference_predict(arm, {'linear_frozen': linear}, batch)
            return predict
        for name in CONFIRM:
            check()
            record = load_confirmation(plan, name)
            np.savez_compressed(output / ('confirm-data-'+name+'.npz'), q=record['q'], u=record['u'], raw_indices=record['raw_indices'])
            batch, physical, starts = make_windows(record, norm, cfg)
            np.savez_compressed(output / ('confirm-windows-'+name+'.npz'), starts=starts, target=batch['target'])
            if physical_first is None:
                physical_first = {k: v[:1].copy() for k, v in physical.items() if k != 'target'}
            identities = [({k: r[k] for k in ('key', 'arm', 'seed', 'learning_rate')}, predictor(r['arm'], r['key'])) for r in model_records]
            identities += [({'key': arm, 'arm': arm, 'seed': None, 'learning_rate': None}, predictor(arm)) for arm in REFERENCES]
            for identity, predict in identities:
                check()
                prediction, error = prediction_attempt(predict, batch)
                filename = 'prediction-'+name+'-'+identity['key']+'.npz'
                if prediction is not None:
                    np.savez_compressed(output / filename, prediction=prediction)
                common = {'recording': name, **identity}
                scored = old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error)
                rows.extend(scored)
                attempt = {**common, 'status': 'PASS' if all(r['status'] == 'PASS' for r in scored) else 'FAILED',
                           'errors': [r['error'] for r in scored], 'prediction_file': filename if prediction is not None else None,
                           'prediction_pin': old.descriptor(output/filename) if prediction is not None else None}
                attempts.append(attempt)
                old.write_json(output/f'completed-prediction-{len(attempts):02d}.json', attempt)
        old.require(len(attempts) == 56 and len(rows) == 112, 'all fixed prediction attempts and horizons retained')
        old.write_json(output / 'prediction-attempts.json', attempts)
        for identity in [*model_records, *[{'key': a, 'arm': a, 'seed': None, 'learning_rate': None} for a in REFERENCES]]:
            check()
            arm, key = identity['arm'], identity['key']
            if arm in ALL_ARMS:
                resource = dict(identity['resources'])
            else:
                count = 445440 if arm in ridges else 150 if arm == 'linear_frozen' else 0
                state = 192 if arm in ridges else 18 if arm == 'linear_frozen' else 6
                resource = {'parameters': count, 'parameter_bytes': count*8, 'buffer_bytes': 16 if arm in ridges else 0,
                            'state_scalars': state, 'state_bytes': state*8, 'normalizer_bytes': 192, 'dtype': 'float64',
                            'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}
            row = {k: identity[k] for k in ('key', 'arm', 'seed', 'learning_rate')}
            row.update(resource)
            row.update(timing_attempt(predictor(arm, key if arm in ALL_ARMS else None), physical_first, norm, cfg))
            resources.append(row)
            old.write_json(output/f'completed-timing-{len(resources):02d}.json', row)
        old.require(len(resources) == 28, 'all28 fixed cost attempts retained')
        after = {r['key']: parameter_hash(models[r['key']]) for r in model_records}
        old.require(after == {r['key']: r['state_sha256'] for r in model_records}
                    and all(p.grad is None and not p.requires_grad for m in models.values() for p in m.parameters()), 'all loaded parameters unchanged without gradients')
        old.write_json(output/'parameter-checks.json', {'before': {r['key']: r['state_sha256'] for r in model_records}, 'after': after,
                       'unchanged': True, 'optimizer_updates': 0})
        result = fixed_rule(rows, resources, cfg)
        old.write_json(output/'results.json', {'version': VERSION, 'config': cfg, 'fixed_rates': dict(FIXED_RATES), 'rows': rows, 'result': result})
        old.write_json(output/'resources.json', resources)
        check()
        final_plan, final_sha = authenticate(registration)
        old.require(final_plan == plan and final_sha == sha, 'unchanged registered source/input evidence')
        check()
        old.write_json(output/'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}})
        check()
        old.write_json(output/'receipt.json', {'status': 'PASS', 'registration_sha256': sha, 'scientific_result': result['status'],
                       'seconds': (clock.now_ns()-started_ns)/1e9, 'monotonic_seconds': time.monotonic()-started,
                       'clock': clock.backend, 'models': 24, 'prediction_attempts': 56, 'rows': 112, 'resource_attempts': 28,
                       'raw_decodes': 2, 'fits': 0, 'optimizer_updates': 0, 'normalizer_refits': 0, 'reference_refits': 0,
                       'recipe_selection_calls': 0, 'confirmation_access': True, 'official_test_access': False})
        return result
    except BaseException as exc:
        old.write_json(output/'failure.json', {'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc),
                       'models_loaded': len(model_records), 'prediction_attempts': len(attempts), 'rows': len(rows),
                       'resource_attempts': len(resources), 'seconds': (clock.now_ns()-started_ns)/1e9})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registration, args.output)
