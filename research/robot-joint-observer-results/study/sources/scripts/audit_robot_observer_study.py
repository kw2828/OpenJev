"""Saved-output observer audit; independent scores/rules and qualified replay.

No training, raw recording decoder, optimizer replay, or scientific generation.
Numeric imports occur only after the original process and opaque files admit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import audit_robot_history_confirmation as prior_audit

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-observer-study-audit-v1'
LEARNED = ('local_affine', 'temporal_affine', 'observer_learned')
FIXED = ('last_two', 'observer_fixed', 'observer_zero')
CACHED = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded',
          'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*LEARNED, *FIXED, *CACHED)
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES = (8101, 8102, 8103), (.001, .003)
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
CONDITIONS = ('complete_forecasts_and_costs', 'equal_four_file_mean_5pct_vs_best_control',
              'each_file_within_2pct_best_simple', 'latency_within_150pct_last_two',
              'complete_frontier_not_dominated')
SOURCES = (*prior_audit.SOURCES, 'scripts/publish_robot_history_confirmation.py',
           'tests/test_publish_robot_history_confirmation.py', 'src/openjev/research/robot_observer_initializer.py',
           'tests/test_robot_observer_initializer.py', 'scripts/robot_observer_study.py',
           'tests/test_robot_observer_study.py', 'research/robot-observer-protocol.md',
           'scripts/audit_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py')
LAUNCHER = 'scripts/launch_robot_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:], LAUNCHER)
DEV = ('recording_2021_12_15_21H_54M.mat', 'recording_2021_12_15_22H_10M.mat')
EXPOSED = (*DEV, *prior_audit.CONFIRM)
CACHED_RATES = {arm: prior_audit.FIXED_RATES[arm.removeprefix('joint_')] for arm in CACHED}
STUDY = 'output/robot-observer-study-v1'
ENGINEERING = 'output/robot-observer-engineering-v1'
REGISTRATION = 'research/robot-observer-registration.json'
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_observer_study.py', '--registration', REGISTRATION, '--output', STUDY]
NUMERIC_ERRORS = (*prior_audit.NUMERIC_ERRORS, 'finite CPU tensor with exact shape/dtype: observer innovation')


def identities():
    result = [{'key': f'{arm}-{seed}-lr{i}', 'arm': arm, 'seed': seed, 'learning_rate': rate, 'origin': 'fresh'}
              for seed in SEEDS for i, rate in enumerate(RATES) for arm in LEARNED]
    result += [{'key': f'{arm}-{seed}-fixed', 'arm': arm, 'seed': seed, 'learning_rate': None, 'origin': 'fixed'}
               for arm in FIXED for seed in SEEDS]
    result += [{'key': f'{arm}-{seed}-cached', 'arm': arm, 'seed': seed, 'learning_rate': CACHED_RATES[arm], 'origin': 'cached'}
               for arm in CACHED for seed in SEEDS]
    return result + [{'key': arm, 'arm': arm, 'seed': None, 'learning_rate': None, 'origin': 'reference'} for arm in REFS]


def expected_config(parent):
    keys = ('context', 'train_horizon', 'dev_horizon', 'horizons', 'skip', 'dev_stride', 'updates', 'batch_size',
            'adam_betas', 'adam_eps', 'gradient_clip', 'window_seed_offset', 'preprocessing_sha256',
            'timing_warmups', 'timing_repeats', 'seeds', 'learning_rates')
    return {**{key: parent[key] for key in keys}, 'version': 'robot-observer-study-v1', 'learned': list(LEARNED),
            'fixed': list(FIXED), 'cached': list(CACHED), 'references': list(REFS), 'families': [*ARMS, *REFS],
            'cached_rates': CACHED_RATES, 'partitions': {'fit': parent['partitions']['fit'], 'dev': list(DEV), 'exposed': list(EXPOSED)},
            'fit_cap_seconds': 1800., 'wall_cap_seconds': 14400., 'mean_reduction': .05, 'file_harm_ratio': 1.02,
            'latency_ratio': 1.5, 'all_evaluation_data_exposed': True, 'official_test_access': False,
            'raw_mat_access': False, 'normalizer_refits': 0, 'reference_refits': 0,
            'selection_scope': 'pooled H128 SSE on original two DEV files only; lower rate wins exact ties'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink file: ' + str(path))
    blob = path.read_bytes()
    return {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}


def pin(path):
    return {'path': str(Path(path).resolve()), **descriptor(path)}


def checked_pin(item, label):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256', 'bytes'}
            and Path(item['path']).is_absolute(), label + ': descriptor')
    require(descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, label + ': bytes')
    return item


def authenticate(study, run_receipt):
    """Original terminal, frozen source, prior audit, and opaque inventory first."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    engineering, registration = ROOT/ENGINEERING, ROOT/REGISTRATION
    require(study == ROOT/STUDY and run_receipt == engineering/'run-process-01.json', 'fixed original study/process')
    plan, sha = read(registration), descriptor(registration)['sha256']
    require(plan['version'] == 'robot-observer-study-v1' and set(plan['sources']) == set(SOURCES), 'exact45 frozen sources')
    for name, value in plan['sources'].items():
        require(descriptor(ROOT/name) == value == descriptor(study/'sources'/name), 'source/snapshot: '+name)
    require(descriptor(study/'registration.json') == descriptor(registration)
            and descriptor(ROOT/LAUNCHER) == plan['launcher'], 'registration/launcher identity')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread environment')
    process, launch = read(run_receipt), read(engineering/'run-launch-01.json')
    keys = {'command', 'prefit_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == keys and set(process) == keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'}
            and all(process[key] == value for key, value in launch.items()), 'original launch/terminal schema and join')
    require(process['command'] == COMMAND and process['registration_sha256'] == sha
            and process['launcher'] == plan['launcher'] and process['thread_env'] == dict.fromkeys(THREADS, '1')
            and type(process['returncode']) is int and process['returncode'] == 0 and process['external_timeout'] is False
            and type(process['elapsed_seconds']) in (int, float) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= 14460
            and descriptor(run_receipt.with_suffix('.log')) == process['log'], 'successful bounded original process')
    commit = process['prefit_commit']
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'prefit commit identity')
    for name in (*SOURCES, LAUNCHER, REGISTRATION):
        require(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) == (ROOT/name).read_bytes(),
                'prefit committed bytes: '+name)
    q = read(checked_pin(plan['qualification'], 'qualification')['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_observer_initializer.py',
                 'tests/test_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py']]
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
            and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(THREADS, '1')
            and [row['command'] for row in q['commands']] == commands, 'exact original qualification')
    require(datetime.fromisoformat(q['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification and registration precede original launch')
    for row in q['commands']:
        require(type(row['returncode']) is int and row['returncode'] == 0
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] > 0
                and descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification log/command')
    # Qualified admission hashes historical raw media opaquely, without decoding.
    # Only the preprocessed NPZ payloads below can reach this auditor's loader.
    import publish_robot_history_confirmation as parent
    prior = parent.authenticate(ROOT/parent.STUDY, ROOT/parent.AUDIT, ROOT/parent.ENGINEERING)
    history_reg = ROOT/'research/robot-history-initialization-registration.json'
    require(descriptor(history_reg)['sha256'] == prior_audit.PARENT_SHA, 'fixed history registration')
    history = read(history_reg)
    require(plan['config'] == expected_config(history['config']), 'literal prospective observer config')
    history_folder = ROOT/'output/robot-history-initialization-study-v1'
    history_inventory = read(history_folder/'manifest.json')['files']
    expected_inputs = {'parent/'+name: item for name, item in prior['plan']['parent_payloads'].items()}
    for name in ('fits.json', *(f'batches-{seed}.npz' for seed in SEEDS)):
        item = pin(history_folder/name)
        require({k: item[k] for k in ('bytes', 'sha256')} == history_inventory[name], 'original history payload')
        expected_inputs['parent/'+name] = item
    for part in ('fit', 'dev'):
        for name in history['config']['partitions'][part]:
            expected_inputs['data/'+name] = history['data'][part+'-data-'+name+'.npz']
    for name in prior_audit.CONFIRM:
        filename = 'confirm-data-'+name+'.npz'
        item = pin(ROOT/parent.STUDY/filename)
        require({k: item[k] for k in ('bytes', 'sha256')} == prior['inventory'][filename], 'already exposed data lineage')
        expected_inputs['data/'+name] = item
    require(len(expected_inputs) == 45 and plan['inputs'] == expected_inputs, 'exact34 parent and11 saved recording inputs')
    for name, item in plan['inputs'].items():
        checked_pin(item, name)
        if name.startswith('parent/'):
            require(descriptor(study/name) == {k: item[k] for k in ('bytes', 'sha256')}, 'byte-identical inherited copy')
    require(plan['parent_publication_manifest'] == pin(ROOT/parent.OUTPUT/'manifest.json')
            and plan['parent_audit'] == pin(ROOT/parent.AUDIT), 'failed confirmation preserved as parent evidence')
    manifest, paths = read(study/'manifest.json'), list(study.rglob('*'))
    require(set(manifest) == {'files'} and not any(path.is_symlink() for path in paths), 'regular complete output inventory')
    require({str(path.relative_to(study)) for path in paths if path.is_file()}
            == set(manifest['files']) | {'manifest.json', 'receipt.json'}, 'no omitted output files')
    for name, expected in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and str(Path(name)) == name
                and descriptor(study/name) == expected, 'opaque output hash: '+name)
    runtime, receipt = read(study/'runtime.json'), read(study/'receipt.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform()
            and runtime['machine'] == platform.machine() and runtime['torch_threads'] == 1
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1'), 'same qualified runtime/platform')
    for package in ('numpy', 'torch'):
        require(importlib.metadata.version(package) == runtime[package], 'same numeric package: '+package)
    require(runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'), 'native suspend clock')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == sha
            and [receipt[key] for key in ('fresh_fits', 'fixed_models', 'cached_models', 'rows', 'prediction_attempts',
                                         'timing_attempts', 'raw_mat_decodes', 'saved_fit_loads', 'saved_exposed_loads')]
            == [18, 9, 21, 416, 208, 43, 0, 7, 4]
            and receipt['all_evaluation_data_exposed'] is True and receipt['official_test_access'] is False
            and receipt['clock'] == runtime['clock'] and 0 < receipt['seconds'] <= 14400
            and 0 < receipt['monotonic_seconds'] <= process['elapsed_seconds'], 'complete producer terminal')
    inputs = {key: pin(path) for key, path in {'manifest': study/'manifest.json', 'producer_receipt': study/'receipt.json',
              'registration': registration, 'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'),
              'run_launch': engineering/'run-launch-01.json', 'runtime': study/'runtime.json', 'launcher': ROOT/LAUNCHER}.items()}
    inputs.update(qualification=plan['qualification'], qualification_logs=[pin(row['log']) for row in q['commands']],
                  parent_admission=prior['inputs'], parent_inputs=plan['inputs'], parent_audit=plan['parent_audit'],
                  parent_publication_manifest=plan['parent_publication_manifest'])
    return plan, inputs


def close(actual, expected, label):
    """Exact identities/schema; tolerance applies only to saved float scalars."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), label + ': keys')
        for name in actual:
            close(actual[name], expected[name], label + '/' + name)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), label + ': list')
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close(left, right, label + '/' + str(i))
    elif type(actual) is float:
        require(type(expected) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), label + ': scalar')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': identity')


def scored(prediction, target, scale, horizon, np):
    """Full saved-bank scoring in standardized and physical units."""
    require(horizon in (64, 128) and prediction.shape == target.shape and target.ndim == 3
            and target.shape[1:] == (128, 6) and len(target) > 0 and scale.shape == (6,)
            and np.isfinite(target).all() and np.isfinite(scale).all() and (scale > 0).all(), 'score geometry')
    if not np.isfinite(prediction).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        delta = prediction[:, :horizon] - target[:, :horizon]
        sse = float(np.sum(delta * delta))
        joint = np.mean((delta * scale) ** 2, axis=(0, 1))
    if not math.isfinite(sse) or not np.isfinite(joint).all():
        return None
    return {'standardized_rmse': math.sqrt(sse / delta.size), 'standardized_sse': sse,
            'scalars': int(delta.size), 'physical_rmse_deg': math.sqrt(float(np.mean(joint))),
            'per_joint_rmse_deg': np.sqrt(joint).tolist(), 'windows': len(target), 'horizon': horizon}


def metric_rows(common, prediction, target, scale, error, np):
    result = []
    for horizon in (64, 128):
        local, value = error, None
        if local is None:
            require(prediction.dtype == np.float64 and prediction.shape == target.shape, 'full float64 forecast bank')
            value = scored(prediction, target, scale, horizon, np)
            if value is None:
                local = {'type': 'NonfiniteEvaluation', 'message': 'nonfinite prediction/target'
                         if not np.isfinite(prediction).all() else 'nonfinite metric arithmetic'}
        result.append({**common, 'horizon': horizon, 'status': 'PASS' if local is None else 'FAILED',
                       'error': local, 'metrics': value})
    return result


def windows(record, norm, np):
    """Causal alignment: u31 predicts q32, independent of producer helpers."""
    require(set(record) == {'q', 'u', 'raw_indices'} and record['q'].shape == record['u'].shape == (3636, 6),
            'complete exposed saved recording')
    require(all(record[k].dtype == np.float64 and np.isfinite(record[k]).all() for k in ('q', 'u'))
            and record['raw_indices'].dtype == np.int64
            and np.array_equal(record['raw_indices'], np.arange(0, 90881, 25, dtype=np.int64)), 'saved physical geometry')
    q = (record['q'] - norm['q_mean']) / norm['q_std']
    u = (record['u'] - norm['u_mean']) / norm['u_std']
    starts = np.arange(64, 3636 - 32 - 128 + 1, 160, dtype=np.int64)
    batch = {'q_context': np.stack([q[s:s+32] for s in starts]),
             'u_context': np.stack([u[s:s+32] for s in starts]),
             'future_u': np.stack([u[s+31:s+159] for s in starts])}
    return starts, batch, np.stack([q[s+32:s+160] for s in starts])


def state_hash(values):
    digest = hashlib.sha256()
    for name, value in values.items():
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes())
    return digest.hexdigest()


def decisions(rows, resources, cfg):
    """DEV2-only selection; four already exposed files decide all five rules."""
    dev, exposed = cfg['partitions']['dev'], cfg['partitions']['exposed']
    cached_rates = cfg['cached_rates']
    require(len(dev) == 2 and len(exposed) == len(set(exposed)) == 4 and set(dev) <= set(exposed)
            and set(cached_rates) == set(CACHED), 'fixed evaluation and cached recipe roster')
    recipes = [(arm, seed, rate) for arm in LEARNED for seed in SEEDS for rate in RATES]
    recipes += [(arm, seed, None) for arm in FIXED for seed in SEEDS]
    recipes += [(arm, seed, cached_rates[arm]) for arm in CACHED for seed in SEEDS]
    recipes += [(arm, None, None) for arm in REFS]
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    expected = {(name, arm, seed, rate, h) for name in exposed for arm, seed, rate in recipes for h in (64, 128)}
    require(len(rows) == len(lookup) == 416 and set(lookup) == expected, 'complete416 duplicate-free metric roster')

    def metric(name, arm, seed, rate, horizon=128):
        row = lookup[name, arm, seed, rate, horizon]
        require(row['status'] in ('PASS', 'FAILED'), 'declared metric status')
        if row['status'] == 'FAILED':
            require(row['metrics'] is None and row['error'] is not None, 'retained numeric failure')
            return None
        value = row['metrics']
        require(row['error'] is None and isinstance(value, dict), 'successful metric record')
        numbers = [value[k] for k in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')]
        numbers += value['per_joint_rmse_deg']
        require(type(value['scalars']) is int and value['scalars'] == 22*horizon*6
                and value['windows'] == 22 and value['horizon'] == horizon and len(value['per_joint_rmse_deg']) == 6
                and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in numbers), 'finite nonnegative scalar metrics')
        return value

    for name, arm, seed, rate, h in expected:
        metric(name, arm, seed, rate, h)
    chosen, options = {}, {}
    for arm in LEARNED:
        candidates, options[arm] = [], []
        for rate in RATES:
            values = [metric(name, arm, seed, rate) for name in dev for seed in SEEDS]
            score = sum(v['standardized_sse'] for v in values) if all(v is not None for v in values) else None
            options[arm].append({'learning_rate': rate, 'complete': score is not None, 'pooled_sse': score})
            if score is not None:
                candidates.append((score, rate))
        chosen[arm] = min(candidates)[1] if candidates else None
    chosen.update({arm: None for arm in FIXED})
    chosen.update(cached_rates)
    selection = {'selected_rates': {arm: chosen[arm] for arm in LEARNED}, 'options': options,
                 'scope': cfg['selection_scope']}
    details, means = {}, {}
    all_selected_finite = all(chosen[arm] is not None for arm in LEARNED)
    for name in exposed:
        local = {}
        for arm in (*ARMS, *REFS):
            if arm in LEARNED and chosen[arm] is None:
                continue
            seeds = SEEDS if arm in ARMS else (None,)
            rate = chosen[arm] if arm in ARMS else None
            values = [metric(name, arm, seed, rate) for seed in seeds]
            both = [metric(name, arm, seed, rate, h) for seed in seeds for h in (64, 128)]
            all_selected_finite &= all(value is not None for value in both)
            if all(value is not None for value in values):
                local[arm] = sum(value['standardized_rmse'] for value in values) / len(values)
        details[name] = {'means': local}
    for arm in (*ARMS, *REFS):
        if all(arm in details[name]['means'] for name in exposed):
            means[arm] = sum(details[name]['means'][arm] for name in exposed) / 4
    expected_costs = {(arm, seed, chosen[arm]) for arm in ARMS for seed in SEEDS}
    expected_costs |= {(arm, None, None) for arm in REFS}
    keys = [(r['arm'], r['seed'], r['learning_rate']) for r in resources]
    require(len(keys) == len(set(keys)) == len(expected_costs) and set(keys) == expected_costs, 'all scheduled selected resource identities')
    for row in resources:
        require(row['status'] in ('PASS', 'FAILED', 'UNAVAILABLE'), 'declared resource status')
        if row['status'] == 'PASS':
            require(row['error'] is None and isinstance(row['timing'], dict), 'successful resource evidence')
        else:
            require(row['error'] is not None and row['timing'] is None, 'failed or unavailable resource evidence')
    costs = {}
    for arm in (*ARMS, *REFS):
        subset = [row for row in resources if row['arm'] == arm]
        costs[arm] = None
        if len(subset) != (3 if arm in ARMS else 1):
            continue
        if any(row['status'] != 'PASS' for row in subset):
            continue
        times = [row['timing']['median_seconds'] for row in subset]
        sizes = [[row[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')] for row in subset]
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in times)
                and all(type(v) is int and v >= 0 for parts in sizes for v in parts), 'valid complete request cost')
        costs[arm] = {'latency': sorted(times)[len(times)//2], 'bytes': max(sum(parts) for parts in sizes)}
    candidate, controls = 'observer_learned', tuple(a for a in (*ARMS, *REFS) if a != 'observer_learned')
    frontier = all(arm in means and costs[arm] is not None for arm in (*ARMS, *REFS))
    complete = all_selected_finite and frontier
    best = min((means[arm] for arm in controls), default=None) if all(arm in means for arm in controls) else None
    gain = candidate in means and best is not None and best > 0 and means[candidate] <= .95*best
    simple = ('last_two', 'local_affine', 'temporal_affine', 'observer_fixed')
    harm = all(all(arm in details[name]['means'] for arm in (candidate, *simple))
               and details[name]['means'][candidate] <= 1.02*min(details[name]['means'][arm] for arm in simple) for name in exposed)
    left, right = costs[candidate], costs['last_two']
    latency = left is not None and right is not None and left['latency'] <= 1.5*right['latency']
    dominators = []
    if frontier:
        point = (means[candidate], left['latency'], left['bytes'])
        for arm in controls:
            control = (means[arm], costs[arm]['latency'], costs[arm]['bytes'])
            if all(a <= b for a, b in zip(control, point, strict=True)) and any(a < b for a, b in zip(control, point, strict=True)):
                dominators.append(arm)
    flags = (complete, gain, harm, latency, frontier and not dominators)
    conditions = [{'name': name, 'passed': bool(flag)} for name, flag in zip(CONDITIONS, flags, strict=True)]
    passed = sum(row['passed'] for row in conditions)
    return {'selection': selection, 'result': {'status': 'OBSERVER_DEVELOPMENT_PASS' if passed == 5 else 'OBSERVER_DEVELOPMENT_FAIL',
                       'passed': passed, 'total': 5, 'conditions': conditions, 'details': details,
                       'equal_file_means': means, 'costs': costs, 'best_control_mean': best, 'frontier_complete': frontier,
                       'dominators': dominators, 'scope': 'All four files exposed development; not confirmation, independent robots, novelty or control'}}


def validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np, updates=4096):
    """Cell byte identity and initializer-only Adam slots, including failed fits."""
    require(mode in (*LEARNED, *FIXED) and set(initial) == set(final), 'observer checkpoint identity')
    cell_keys = {name for name in initial if name.startswith('cell.')}
    require(cell_keys == set(backbone) and sum(v.size for v in backbone.values()) == 590, 'exact590 inherited cell')
    for name, original in backbone.items():
        require(original.dtype == initial[name].dtype == final[name].dtype == np.float32
                and original.shape == initial[name].shape == final[name].shape
                and np.isfinite(original).all() and initial[name].tobytes() == original.tobytes()
                and final[name].tobytes() == original.tobytes(), 'immutable inherited cell: ' + name)
    extra = {'head.weight': (12, 30), 'head.bias': (12,)} if mode in LEARNED[:2] else (
        {'gain': (12, 6)} if mode in ('observer_learned', 'observer_fixed', 'observer_zero') else {})
    require(set(initial) - cell_keys == set(extra), 'exact initializer keys')
    for name, shape in extra.items():
        require(initial[name].shape == final[name].shape == shape and initial[name].dtype == final[name].dtype == np.float32
                and np.isfinite(initial[name]).all(), 'initializer shape/dtype')
        desired = np.zeros(shape, dtype=np.float32)
        if name == 'gain' and mode != 'observer_zero':
            desired = np.tile(np.eye(6, dtype=np.float32), (2, 1))
        require(np.array_equal(initial[name], desired), 'declared zero-head or identity-gain initialization')
        if mode in FIXED:
            require(final[name].tobytes() == desired.tobytes(), 'fixed gain unchanged')
    shapes = extra if mode in LEARNED else {}
    permitted = {name + '/' + suffix for name in shapes for suffix in ('exp_avg', 'exp_avg_sq', 'step')}
    require(set(optimizer) <= permitted and (not optimizer or set(optimizer) == permitted), 'initializer-only complete Adam ownership')
    steps = receipt['completed_updates']
    require(type(steps) is int and 0 <= steps <= updates and receipt['status'] in ('PASS', 'FAILED'), 'update count/status')
    require(mode not in FIXED or (steps == 0 and not optimizer), 'fixed models never have optimizer updates')
    require(not steps or optimizer, 'completed updates require Adam evidence')
    for name, shape in shapes.items():
        if optimizer:
            for suffix in ('exp_avg', 'exp_avg_sq'):
                value = optimizer[name + '/' + suffix]
                require(value.shape == shape and value.dtype == np.float32, 'Adam moment geometry')
            step = optimizer[name + '/step']
            require(step.shape == () and step.dtype == np.float32 and np.isfinite(step)
                    and float(step) in (steps, steps + 1), 'Adam attempted update ledger')
    if receipt['status'] == 'PASS':
        expected_steps = updates if mode in LEARNED else 0
        require(steps == expected_steps and all(np.isfinite(v).all() for v in (*final.values(), *optimizer.values())),
                'finite complete terminal evidence')
        if shapes:
            require(set(optimizer) == permitted and all(float(optimizer[name+'/step']) == updates for name in shapes),
                    'complete initializer Adam steps')
    return {'common_cell_sha256': state_hash({name: initial[name] for name in initial if name in cell_keys}),
            'frozen_parameters': 590, 'trainable_parameters': sum(math.prod(s) for s in shapes.values()),
            'fixed_gain_scalars': 72 if mode in ('observer_fixed', 'observer_zero') else 0}


def validate_fit_receipt(receipt, trace, rate, cap):
    require(set(receipt) == {'status', 'error', 'completed_updates', 'requested_updates', 'learning_rate',
                            'optimizer_seconds', 'fit_seconds', 'fit_cap_scope', 'timing_scope', 'files'}, 'fit receipt schema')
    require(receipt['status'] in ('PASS', 'FAILED') and type(receipt['completed_updates']) is int
            and len(trace) == receipt['completed_updates'] <= 4096 and receipt['requested_updates'] == 4096
            and receipt['learning_rate'] == rate, 'complete fresh update ledger')
    for update, row in enumerate(trace, 1):
        require(set(row) == {'update', 'loss', 'gradient_norm_before_clip'} and type(row['update']) is int
                and row['update'] == update and all(type(row[k]) in (int, float) and math.isfinite(row[k])
                and row[k] >= 0 for k in ('loss', 'gradient_norm_before_clip')), 'finite completed trace')
    require(all(type(receipt[k]) in (int, float) and math.isfinite(receipt[k]) for k in ('optimizer_seconds', 'fit_seconds'))
            and 0 <= receipt['optimizer_seconds'] <= receipt['fit_seconds'] and receipt['fit_seconds'] > 0, 'fit timing scope')
    require(receipt['fit_cap_scope'] == 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows'
            and receipt['timing_scope'] == 'optimizer loop including batch construction and finite checks, excluding model construction and saved files'
            and set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json'}, 'fit timing and evidence schema')
    if receipt['status'] == 'PASS':
        require(len(trace) == 4096 and receipt['error'] is None and receipt['fit_seconds'] <= cap, 'complete successful fit')
    else:
        error = receipt['error']
        allowed = (*NUMERIC_ERRORS, 'single-fit wall cap exceeded', 'single-fit wall cap exceeded during preservation',
                   'native single-fit deadline exceeded', 'nonfinite autoregressive training loss',
                   'nonfinite training gradient norm', 'nonfinite updated parameters', 'nonfinite updated Adam state')
        require(isinstance(error, dict) and set(error) == {'type', 'message'} and error['type'] == 'FitFailure'
                and error['message'] in allowed, 'declared preserved numerical/deadline failure')


def storage(arm):
    if arm in (*CACHED, *REFS):
        return prior_audit.storage(arm.removeprefix('joint_'))
    require(arm in (*LEARNED, *FIXED), 'declared new model')
    count = 962 if arm in LEARNED[:2] else 662 if arm == 'observer_learned' else 590
    return {'parameters': count, 'parameter_bytes': count*4,
            'buffer_bytes': 288 if arm in ('observer_fixed', 'observer_zero') else 0,
            'state_scalars': 12, 'state_bytes': 48, 'normalizer_bytes': 192, 'inactive_parameters': 0,
            'dtype': 'float32', 'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}


def reference_prediction(arm, coefficients, batch, np):
    value = prior_audit.reference_prediction(arm, coefficients, batch, np)
    # The qualified causal ridge refuses nonfinite output before returning a
    # bank. Linear/persistence retain their returned banks for metric handling.
    if arm in REFS[:2] and not np.isfinite(value).all():
        raise ValueError('nonfinite ridge prediction')
    return value


def resource_identities(selection):
    rows = [row for row in identities() if row['arm'] not in LEARNED
            or row['learning_rate'] == selection['selected_rates'][row['arm']]]
    for arm in LEARNED:
        if selection['selected_rates'][arm] is None:
            rows += [{'key': f'{arm}-{seed}-unavailable', 'arm': arm, 'seed': seed,
                      'learning_rate': None, 'origin': 'unavailable'} for seed in SEEDS]
    require(len(rows) == 43, 'all43 resource attempts')
    return rows


def validate_resources(resources, selection, np):
    require(len(resources) == 43, 'all43 retained request cost rows')
    for row, identity in zip(resources, resource_identities(selection), strict=True):
        spec = storage(identity['arm'])
        require(set(row) == set(identity) | set(spec) | {'status', 'error', 'timing'}
                and all(row[k] == v for k, v in identity.items()), 'exact ordered resource identity/schema')
        close({k: row[k] for k in spec}, spec, 'actual numeric storage')
        if identity['origin'] == 'unavailable':
            require(row['status'] == 'UNAVAILABLE' and row['timing'] is None
                    and row['error'] == {'type': 'UnavailableSelectedRecipe'}, 'explicit unavailable recipe cost')
        elif row['status'] == 'FAILED':
            require(row['timing'] is None and row['error'] in [{'type': 'NonfiniteTiming', 'message': message}
                    for message in (*NUMERIC_ERRORS, 'finite complete timed request', 'nonfinite ridge prediction')], 'known failed timing')
        else:
            require(row['status'] == 'PASS' and row['error'] is None, 'declared successful timing')
            timing = row['timing']
            require(set(timing) == {'seconds', 'median_seconds', 'p95_seconds', 'scope'}
                    and timing['scope'] == prior_audit.TIMING_SCOPE and len(timing['seconds']) == 20
                    and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in timing['seconds']), 'twenty original full-request samples')
            close(timing['median_seconds'], float(np.median(timing['seconds'])), 'saved median')
            close(timing['p95_seconds'], float(np.percentile(timing['seconds'], 95)), 'saved p95')


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_observer_study as replay
    import torch
    torch.set_num_threads(1)
    cfg, roster = plan['config'], identities()
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'qualified_model_replays': 0, 'independent_reference_replays': 0,
              'raw_mat_decodes': 0, 'official_test_decodes': 0, 'optimizer_updates': 0, 'timing_replays': 0}
    expected = {'registration.json', 'runtime.json', 'fits.json', 'checkpoint-barrier.json', 'parameter-checks.json',
                'prediction-attempts.json', 'resources.json', 'results.json', 'parent-parity.json'}
    expected |= {'sources/'+name for name in SOURCES}
    expected |= {name for name in plan['inputs'] if name.startswith('parent/')}

    def load(name, external=False):
        if external:
            require(name in plan['inputs'] and name.startswith('data/'), 'only11 saved physical recordings')
            path = checked_pin(plan['inputs'][name], name)['path']
        else:
            expected.add(name)
            path = study/name
        with np.load(path, allow_pickle=False) as bank:
            values = {name: bank[name].copy(order='K') for name in bank.files}
        counts['npz_decodes'] += 1
        counts['array_decodes'] += len(values)
        return values

    def equal(actual, wanted, label):
        require(actual.dtype == wanted.dtype and actual.shape == wanted.shape
                and np.array_equal(actual, wanted, equal_nan=True), label)

    norm = load('parent/normalizers.npz')
    require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'} and all(v.dtype == np.float64 and v.shape == (6,)
            and np.isfinite(v).all() for v in norm.values()) and all((norm[k] > 0).all() for k in ('q_std', 'u_std')),
            'finite inherited normalization')
    fit_data = [load('data/'+name, external=True) for name in cfg['partitions']['fit']]
    require(len(fit_data) == 7, 'seven inherited FIT recordings')
    for data in fit_data:
        windows(data, norm, np)  # Independent shape, chronological raw-index and finite guards only.
    for field in ('q', 'u'):
        combined = np.concatenate([data[field][64:] for data in fit_data])
        equal(norm[field+'_mean'], combined.mean(0), 'unchanged FIT mean')
        equal(norm[field+'_std'], combined.std(0), 'unchanged FIT population scale')
    batch_banks = {}
    for seed in SEEDS:
        bank = load(f'parent/batches-{seed}.npz')
        batch_banks[seed] = bank
        require(set(bank) == {'record', 'start'}, 'saved batch schema')
        rng = np.random.Generator(np.random.PCG64(seed+520000))
        recording = rng.integers(0, 7, size=(4096, 16), dtype=np.int64)
        starts = np.empty_like(recording)
        for index in np.ndindex(recording.shape):
            starts[index] = rng.integers(64, 3636-32-128+1)
        equal(bank['record'], recording, 'paired original recording draws')
        equal(bank['start'], starts, 'paired original window draws')
    linear_bank = load('parent/linear.npz')
    require(set(linear_bank) == {'coefficient'}, 'linear coefficient bank')
    linear = linear_bank['coefficient']
    require(linear.shape == (6, 25) and linear.dtype == np.float64 and np.isfinite(linear).all(), 'finite inherited linear coefficients')
    coefficients = {'linear_frozen': linear}
    for arm in REFS[:2]:
        values = load('parent/'+arm+'.npz')
        require(set(values) == {f'h{h:03d}' for h in range(1, 129)}, 'complete causal reference bank')
        for h in range(1, 129):
            v = values[f'h{h:03d}']
            require(v.shape == (6, 193+6*h) and v.dtype == np.float64 and np.isfinite(v).all(), 'causal reference geometry')
        coefficients[arm] = values
    backbones = {seed: load(f'parent/last_two-{seed}-lr0/final.npz') for seed in SEEDS}
    ledger = read(study/'parent/fits.json')
    require(len(ledger) == len({r['key'] for r in ledger}) == 48, 'complete inherited historical ledger')
    ledger = {r['key']: r for r in ledger}
    fits = read(study/'fits.json')
    require(len(fits) == 48, 'all18 fresh9 fixed21 cached models')
    models, states, passed_models = {}, {}, set()
    for number, (fit, identity) in enumerate(zip(fits, roster[:48], strict=True), 1):
        arm, seed, key = identity['arm'], identity['seed'], identity['key']
        require(all(fit[k] == value for k, value in identity.items()), 'ordered model identity')
        model = replay.model_for(arm, seed, linear)
        final = load(key+'/final.npz')
        template = model.state_dict()
        require(set(final) == set(template) and all(final[n].dtype == np.float32 and final[n].shape == tuple(v.shape)
                for n, v in template.items()), 'qualified checkpoint shape/key/dtype')
        if arm in (*LEARNED, *FIXED):
            backbone = backbones[seed]
            require(set(backbone) == {'cell.'+n for n in model.cell.state_dict()}, 'exact inherited common cell names')
            model.cell.load_state_dict({n: torch.from_numpy(backbone['cell.'+n]) for n in model.cell.state_dict()}, strict=True)
            digest = state_hash({n.removeprefix('cell.'): v for n, v in backbone.items()})
            require(fit['frozen_cell_sha256'] == digest, 'independent frozen cell hash')
            constructed = {n: value.detach().numpy().copy(order='K') for n, value in model.state_dict().items()}
            if arm in LEARNED:
                expected |= {f'completed-fit-{number:02d}.json', key+'/trace.json', key+'/fit-receipt.json'}
                require(read(study/f'completed-fit-{number:02d}.json') == fit, 'completed fresh attempt receipt')
                initial, optimizer = load(key+'/initial.npz'), load(key+'/optimizer.npz')
                receipt, trace = read(study/key/'fit-receipt.json'), read(study/key/'trace.json')
                require(receipt == fit['fit'], 'fit ledger joins original helper receipt')
                validate_fit_receipt(receipt, trace, identity['learning_rate'], cfg['fit_cap_seconds'])
                for name, item in receipt['files'].items():
                    require(descriptor(study/key/name) == item, 'complete saved fit evidence pin')
                require(set(initial) == set(constructed), 'complete initial checkpoint keys')
                for name in initial:
                    equal(initial[name], constructed[name], 'exact qualified initial state and inherited backbone')
                validate_frozen_evidence(initial, final, optimizer, backbone, arm, receipt, np)
                require(type(fit['native_fit_seconds']) in (float, int) and math.isfinite(fit['native_fit_seconds'])
                        and fit['native_fit_seconds'] > 0 and fit['effective_status'] == ('FAILED'
                        if fit['native_fit_seconds'] >= 1800 else receipt['status']), 'native preservation deadline status')
                fields = set(identity) | {'fit', 'effective_status', 'native_fit_seconds', 'frozen_cell_sha256', 'resources'}
            else:
                validate_frozen_evidence(constructed, final, {}, backbone, arm,
                                         {'status': 'PASS', 'completed_updates': 0}, np)
                for name, value in constructed.items():
                    equal(final[name], value, 'fixed model exact state')
                require(fit['effective_status'] == 'PASS', 'fixed model status')
                fields = set(identity) | {'effective_status', 'frozen_cell_sha256', 'resources'}
        else:
            source = prior_audit.model_key(arm.removeprefix('joint_'), seed)
            require(fit['parent_fit'] == ledger[source] and ledger[source]['fit']['status']
                    == ledger[source]['effective_status'] == fit['effective_status'] == 'PASS', 'unchanged cached historical fit')
            require(descriptor(study/key/'final.npz') == descriptor(study/'parent'/source/'final.npz'), 'byte-identical cached final')
            fields = set(identity) | {'effective_status', 'parent_fit', 'resources'}
        require(set(fit) == fields, 'exact fit/model metadata fields')
        close(fit['resources'], storage(arm), 'all model storage')
        states[key] = state_hash(final)
        if fit['effective_status'] == 'PASS':
            require(all(np.isfinite(v).all() for v in final.values()), 'finite successful final checkpoint')
            model.load_state_dict({n: torch.from_numpy(v) for n, v in final.items()}, strict=True)
            model.eval()
            require(all(p.grad is None for p in model.parameters()), 'no retained gradient arrays')
            passed_models.add(key)
            models[key] = model
    barrier = {'fresh_fit_attempts': 18, 'fixed_models': 9, 'cached_models': 21, 'exposed_loads_this_run': 0,
               'all_evaluation_data_exposed': True, 'state_sha256': states,
               'checkpoints': {row['key']: descriptor(study/row['key']/'final.npz') for row in fits}}
    require(read(study/'checkpoint-barrier.json') == barrier, 'all48 complete checkpoint barrier before exposed decoding')
    require(read(study/'parameter-checks.json') == {'before': states, 'after': states, 'unchanged': True}, 'immutable model inference receipts')
    parity = []
    for seed in SEEDS:
        bank = batch_banks[seed]
        batch = {name: [] for name in ('q_context', 'u_context', 'future_u')}
        for recording, start in zip(bank['record'][0], bank['start'][0], strict=True):
            source = fit_data[int(recording)]
            q = (source['q']-norm['q_mean'])/norm['q_std']
            u = (source['u']-norm['u_mean'])/norm['u_std']
            batch['q_context'].append(q[start:start+32]); batch['u_context'].append(u[start:start+32])
            batch['future_u'].append(u[start+31:start+159])
        batch = {key: np.stack(value) for key, value in batch.items()}
        original = replay.history.model_for('last_two', seed, linear)
        original.load_state_dict({name: torch.from_numpy(value) for name, value in backbones[seed].items()}, strict=True)
        with torch.no_grad():
            equal(replay.old.infer(models[f'last_two-{seed}-fixed'], batch).numpy(),
                  replay.old.infer(original, batch).numpy(), 'last-two exact inherited first FIT forecast')
        digest = state_hash({name.removeprefix('cell.'): value for name, value in backbones[seed].items()})
        parity.append({'seed': seed, 'frozen_cell_sha256': digest, 'exact_parent_forecast': True,
                       'scope': 'first paired FIT batch only; no evaluation data access'})
    require(read(study/'parent-parity.json') == parity, 'all3 inherited last-two parity receipts')
    rows, attempts = [], []
    fit_lookup = {r['key']: r for r in fits}
    for name in EXPOSED:
        starts, batch, target = windows(load('data/'+name, external=True), norm, np)
        target_bank = load('dev-windows-'+name+'.npz')
        require(set(target_bank) == {'starts', 'target'}, 'saved target-window schema')
        equal(target_bank['starts'], starts, 'all22 scheduled start indices')
        equal(target_bank['target'], target, 'independent torque/target alignment')
        for identity in roster:
            arm, key = identity['arm'], identity['key']
            common = {'recording': name, 'arm': arm, 'seed': identity['seed'],
                      'learning_rate': identity['learning_rate'], 'fit_key': key}
            prediction, error = None, None
            filename = 'prediction-'+name+'-'+key+'.npz'
            if arm in ARMS and key not in passed_models:
                error = {'type': 'FailedTrainingAttempt', 'effective_status': fit_lookup[key]['effective_status']}
            else:
                try:
                    if arm in ARMS:
                        counts['qualified_model_replays'] += 1
                        with torch.no_grad():
                            generated = replay.old.infer(models[key], batch).numpy().astype(np.float64)
                    else:
                        counts['independent_reference_replays'] += 1
                        generated = reference_prediction(arm, coefficients, batch, np)
                except replay.old.FitFailure as exc:
                    require(str(exc) in NUMERIC_ERRORS, 'known qualified forecast failure')
                    error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                except ValueError as exc:
                    if str(exc) != 'nonfinite ridge prediction':
                        raise
                    error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                else:
                    bank = load(filename)
                    require(set(bank) == {'prediction'}, 'one saved forecast field')
                    prediction = bank['prediction']
                    equal(prediction, generated, 'exact qualified model or independent reference replay')
            if prediction is None:
                require(not (study/filename).exists(), 'unavailable forecast cannot have hidden bank')
            group = metric_rows(common, prediction, target, norm['q_std'], error, np)
            rows.extend(group)
            attempt = {**common, 'status': 'PASS' if all(r['status'] == 'PASS' for r in group) else 'FAILED',
                       'errors': [r['error'] for r in group], 'prediction_file': filename if prediction is not None else None}
            attempts.append(attempt)
            completed = f'completed-prediction-{len(attempts):03d}.json'; expected.add(completed)
            close(read(study/completed), attempt, 'every original prediction attempt')
    close(read(study/'prediction-attempts.json'), attempts, 'all208 retained forecast attempts')
    resources = read(study/'resources.json')
    decision = decisions(rows, resources, cfg)
    validate_resources(resources, decision['selection'], np)
    for index, resource in enumerate(resources, 1):
        name = f'completed-timing-{index:02d}.json'; expected.add(name)
        require(read(study/name) == resource, 'every original timing receipt')
    results = {'version': cfg['version'], 'config': cfg, 'selection': decision['selection'], 'rows': rows, 'result': decision['result']}
    close(read(study/'results.json'), results, 'independent metrics, selection, and all five rules')
    require(read(study/'receipt.json')['scientific_result'] == decision['result']['status'], 'terminal scientific outcome')
    require(set(read(study/'manifest.json')['files']) == expected, 'complete exact dynamic evidence roster')
    require(all(state_hash({n: v.detach().numpy() for n, v in model.state_dict().items()}) == states[key]
                for key, model in models.items()), 'replay cannot mutate learned state')
    require(authenticate(study, run_receipt) == (plan, inputs), 'unchanged original admitted evidence')
    counts.update(fresh_fits=18, fixed_models=9, cached_models=21, final_checkpoints=48, initial_checkpoints=18,
                  optimizer_checkpoints=18, metric_rows=416, prediction_attempts=208,
                  prediction_files=sum(a['prediction_file'] is not None for a in attempts), resource_rows=43,
                  completed_fresh_updates=sum(r['fit']['completed_updates'] for r in fits if r['origin'] == 'fresh'),
                  requested_fresh_updates=73728, failed_fresh_fits=sum(r['effective_status'] != 'PASS' for r in fits[:18]),
                  failed_metric_rows=sum(r['status'] != 'PASS' for r in rows), manifest_files=len(expected),
                  frozen_cell_checks=27, parent_parity_model_calls=6, saved_fit_recordings=7, saved_exposed_recordings=4, condition_rows=5)
    return {'version': VERSION, 'status': 'PASS', 'agreement': True, 'study': str(study),
            'registration_sha256': inputs['registration']['sha256'], 'source_pins': plan['sources'], 'auditor': pin(__file__),
            'inputs': inputs, 'results': results, 'resources': resources, 'prediction_attempts': attempts,
            'counts': counts, 'seconds': time.monotonic()-started,
            'checks': {'original_closed_process': True, 'independent_saved_metrics_and_rules': True,
                       'selection_original_dev2_only': True, 'all590_backbone_values_immutable': True,
                       'initializer_only_adam_state': True, 'exact_qualified_model_replay': True,
                       'independent_causal_reference_replay': True, 'all_failed_attempts_retained': True},
            'scope': ['No raw MAT numerical decoding, new fitting, optimizer replay, or official TEST access.',
                      'Historical raw media are hashed opaquely only through the qualified parent admission.',
                      'Model recurrence replay uses qualified production implementations; metric/rule/reference arithmetic is independent.',
                      'Saved training traces and timing samples are checked structurally, never rerun.',
                      'All four evaluation recordings were already exposed; this is development, not confirmation.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--run-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.parent.exists(), 'exclusive audit output directory')
    result = audit(args.study, args.run_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    with (args.output.parent/'manifest.json').open('x') as handle:
        json.dump({'files': {args.output.name: descriptor(args.output)}}, handle, indent=2, sort_keys=True); handle.write('\n')
    print(json.dumps({'status': result['status'], 'agreement': result['agreement'], 'counts': result['counts'],
                      'scientific_status': result['results']['result']['status']}), flush=True)


if __name__ == '__main__':
    main()
