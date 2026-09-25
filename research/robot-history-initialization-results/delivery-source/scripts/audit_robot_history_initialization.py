"""Saved-output history audit: independent scores/rules, qualified model replay.

No raw recordings, training, optimizer calls, or new scientific generation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = '628b038c910bf37246e939b104d70530f6783a7fbc67f79afeb79b8813738994'
COMMIT = '80fc2369b60d8b8434832af904a1f50c099f8031'
FRESH_ARMS = ('last_two', 'local_affine', 'temporal_affine')
CACHED_ARMS = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*FRESH_ARMS, *CACHED_ARMS)
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES = (8101, 8102, 8103), (.001, .003)
PARAMETERS = dict(zip(ARMS, (590, 962, 962, 806, 806, 5916, 1014, 1296), strict=True))
PERMUTATION = (*range(29, -1, -1), 30, 31)
LEGACY_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')
COMMON_PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                   'causal_ridge_100.npz', 'causal_ridge_100.json',
                   *(f'batches-{seed}.npz' for seed in SEEDS))
PAYLOADS = {
    'structured': (*COMMON_PAYLOADS, 'fits.json', 'results.json',
                   *(f'{arm}-{seed}-lr{ri}/{name}' for arm in CACHED_ARMS[:-1]
                     for seed in SEEDS for ri in range(2) for name in LEGACY_FILES)),
    'transition': ('fits.json', 'results.json',
                   *(f'gru_residual-{seed}-lr{ri}/{name}' for seed in SEEDS for ri in range(2)
                     for name in LEGACY_FILES)),
}
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_history_initialization_study.py', '--registration',
           'research/robot-history-initialization-registration.json', '--output', 'output/robot-history-initialization-study-v1']
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py', 'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py', 'tests/test_industrial_robot_data.py', 'tests/test_robot_coupling_study.py', 'research/robot-coupling-protocol.md', 'src/openjev/research/bounded_robot_transition.py', 'src/openjev/research/causal_robot_ridge.py', 'scripts/robot_transition_study.py', 'tests/test_bounded_robot_transition.py', 'tests/test_causal_robot_ridge.py', 'tests/test_robot_transition_study.py', 'research/robot-transition-protocol.md', 'src/openjev/research/compact_robot_gate.py', 'tests/test_compact_robot_gate.py', 'src/openjev/research/structured_robot_transition.py', 'tests/test_structured_robot_transition.py', 'scripts/robot_structured_study.py', 'tests/test_robot_structured_study.py', 'research/robot-structured-protocol.md', 'src/openjev/research/robot_history_initializer.py', 'tests/test_robot_history_initializer.py', 'scripts/robot_history_initialization_study.py', 'tests/test_robot_history_initialization_study.py', 'research/robot-history-initialization-protocol.md', 'scripts/plot_robot_structured.py', 'scripts/audit_robot_structured.py', 'src/openjev/research/suspend_clock.py')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular nonsymlink file: ' + str(path))
    value = path.read_bytes()
    return {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}

def pin(path):
    return {'path': str(Path(path).resolve()), **descriptor(path)}

def close(a, b, name):
    """Schema exact; saved-scalar roundoff only, never checkpoint tolerance."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), name + ': keys')
        for key in a:
            close(a[key], b[key], name + '/' + key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), name + ': list')
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            close(left, right, name + '/' + str(i))
    elif isinstance(a, float):
        require(type(b) in (float, int) and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), name + ': scalar')
    else:
        require(type(a) is type(b) and a == b, name + ': exact')

def checked_pin(item, label):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256', 'bytes'}
            and Path(item['path']).is_absolute(), label + ': descriptor')
    require(descriptor(item['path']) == {k: item[k] for k in ('sha256', 'bytes')}, label + ': bytes')
    return item


def copied_payload(group, name):
    if name in ('fits.json', 'results.json'):
        return f'parent-{group}-{name}'
    return name.replace('gru_residual-', 'gru10-', 1) if group == 'transition' else name


def authenticate(study, run_receipt):
    """Original successful closure and complete opaque provenance before decode."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    engineering = ROOT / 'output/robot-history-initialization-engineering-v1'
    registration = ROOT / 'research/robot-history-initialization-registration.json'
    require(study == ROOT / 'output/robot-history-initialization-study-v1'
            and run_receipt == engineering / 'run-process-01.json', 'fixed original study/process')
    require(COMMIT != 'UNBOUND' and descriptor(registration)['sha256'] == PLAN_SHA, 'frozen registration/commit')
    plan = read(registration)
    require(plan['version'] == 'robot-history-initialization-study-v1' and set(plan['sources']) == set(SOURCES), '29 frozen sources')
    for name, value in plan['sources'].items():
        require(descriptor(ROOT / name) == value == descriptor(study / 'sources' / name), 'source/snapshot: ' + name)
    require(descriptor(study / 'registration.json') == descriptor(registration), 'exact saved registration')
    import audit_robot_structured as parent_admission
    prior, prior_inputs = parent_admission.authenticate(ROOT / 'output/robot-structured-study-v1',
                          ROOT / 'output/robot-structured-engineering-v1/run-process-01.json')
    require(set(prior['sources']) == set(SOURCES[:21])
            and all(prior['sources'][n] == plan['sources'][n] for n in SOURCES[:21]), 'unchanged parent implementations')
    cfg = dict(prior['config'])
    for key in ('mean_ratio', 'paired_ratio', 'storage_ratio', 'joint_harm_ratio', 'direct_noninferiority_ratio'):
        cfg.pop(key, None)
    cfg.update(version='robot-history-initialization-study-v1', arms=list(FRESH_ARMS), comparison_arms=list(ARMS),
               fit_cap_seconds=1800., wall_cap_seconds=14400., latency_ratio=1.25, mean_reduction=.05,
               file_harm_ratio=1.02, cached_parent_refit=False, older_permutation=list(PERMUTATION), diagnostic_gate=False)
    require(plan['config'] == cfg, 'exact prospective config')
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread environment')
    launcher = ROOT / 'scripts/launch_robot_history_initialization.py'
    require(descriptor(launcher) == plan['launcher'], 'launcher source pin')
    process, launch = read(run_receipt), read(engineering / 'run-launch-01.json')
    keys = {'command', 'prefit_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == keys and set(process) == keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'}, 'original closure schema')
    require({k: process[k] for k in keys} == launch and process['command'] == COMMAND
            and process['prefit_commit'] == COMMIT and process['registration_sha256'] == PLAN_SHA
            and process['launcher'] == plan['launcher'] and process['thread_env'] == dict.fromkeys(THREADS, '1'), 'original launch/terminal joins')
    require(type(process['returncode']) is int and process['returncode'] == 0 and process['external_timeout'] is False
            and 0 < process['elapsed_seconds'] <= 14460 and descriptor(run_receipt.with_suffix('.log')) == process['log'], 'successful original process')
    inputs = {key: pin(path) for key, path in {'manifest': study / 'manifest.json', 'producer_receipt': study / 'receipt.json',
              'registration': registration, 'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'),
              'run_launch': engineering / 'run-launch-01.json', 'launcher': launcher, 'runtime': study / 'runtime.json'}.items()}
    runtime = read(study / 'runtime.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform()
            and runtime['machine'] == platform.machine() and runtime['torch_threads'] == 1
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1'), 'same runtime/platform')
    for package in ('numpy', 'torch'):
        require(importlib.metadata.version(package) == runtime[package], 'same package: ' + package)
    require(runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'), 'native suspend clock')
    require(set(plan['parent_closure']) == set(plan['parent_payloads']) == set(plan['parent_registration_sha256'])
            == {'structured', 'transition'}, 'both original parents')
    for group in ('structured', 'transition'):
        folder, eng = ROOT / f'output/robot-{group}-study-v1', ROOT / f'output/robot-{group}-engineering-v1'
        closure = plan['parent_closure'][group]
        paths = {'manifest': folder / 'manifest.json', 'receipt': folder / 'receipt.json', 'process': eng / 'run-process-01.json',
                 'audit': ROOT / f'output/robot-{group}-audit-v1/audit.json', 'audit_process': eng / 'audit-process-01.json'}
        require(set(closure) == set(paths), 'complete parent closure')
        parent = {k: read(checked_pin(v, group + '/' + k)['path']) for k, v in closure.items()}
        require(all(Path(closure[k]['path']) == path for k, path in paths.items()), 'canonical parent closure paths')
        parent_sha = descriptor(ROOT / f'research/robot-{group}-registration.json')['sha256']
        require(parent_sha == plan['parent_registration_sha256'][group]
                == parent['receipt']['registration_sha256'] == parent['audit']['registration_sha256']
                == parent['process']['registration_sha256'], 'parent registration joins')
        require(parent['receipt']['status'] == parent['audit']['status'] == 'PASS' and parent['audit']['agreement'] is True
                and parent['process']['returncode'] == parent['audit_process']['returncode'] == 0, 'parent execution/audit success')
        for key, source in (('manifest', 'manifest'), ('producer_receipt', 'receipt'), ('run_receipt', 'process')):
            require(parent['audit']['inputs'][key] == closure[source], 'parent audit input: ' + key)
        require(parent['audit_process']['audit_output'] == {k: closure['audit'][k] for k in ('sha256', 'bytes')}, 'parent audit terminal')
        if group == 'structured':
            require(parent['audit']['inputs'] == prior_inputs and parent['audit']['source_pins'] == prior['sources']
                    and parent['audit']['auditor'] == pin(ROOT / 'scripts/audit_robot_structured.py'), 'qualified parent audit')
        else:
            require(closure == prior['parent_closure'] and parent_sha == prior['parent_registration_sha256'], 'exact transition ancestor')
        payloads = plan['parent_payloads'][group]
        require(set(payloads) == set(PAYLOADS[group]), 'complete inherited payload roster')
        for name, item in payloads.items():
            checked_pin(item, group + '/payload/' + name)
            expected = {k: item[k] for k in ('sha256', 'bytes')}
            require(Path(item['path']) == folder / name and expected == parent['manifest']['files'][name]
                    == descriptor(study / copied_payload(group, name)), 'original and copied payload: ' + name)
    require(sum(map(len, PAYLOADS.values())) == 163 and plan['data'] == prior['data'] == prior_inputs['parent_data'], '163 payloads/unchanged saved data')
    for name, item in plan['data'].items():
        checked_pin(item, 'saved data/' + name)
    qualification = read(checked_pin(plan['qualification'], 'qualification')['path'])
    require(Path(plan['qualification']['path']) == engineering / 'qualification-02.json'
            and qualification['status'] == 'PASS' and qualification['sources'] == plan['sources']
            and qualification['launcher'] == plan['launcher'] and qualification['sources_unchanged'] is True
            and qualification['thread_env'] == dict.fromkeys(THREADS, '1'), 'exact successful qualification02')
    require(datetime.fromisoformat(qualification['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification/freeze before original fit')
    commands = [['.venv/bin/ruff', 'check', 'src/openjev/research/robot_history_initializer.py', 'tests/test_robot_history_initializer.py',
                 'scripts/robot_history_initialization_study.py', 'tests/test_robot_history_initialization_study.py', 'scripts/launch_robot_history_initialization.py'],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_history_initializer.py', 'tests/test_robot_history_initialization_study.py']]
    qlogs = []
    require(len(qualification['commands']) == 2, 'both qualification commands')
    for i, (row, command) in enumerate(zip(qualification['commands'], commands, strict=True), 1):
        require(row['command'] == command and row['returncode'] == 0 and row['external_timeout'] is False
                and Path(row['log']) == engineering / f'qualification-02/command-{i}.log' and row['seconds'] > 0
                and math.isfinite(row['seconds']) and descriptor(row['log'])['sha256'] == row['sha256'], 'qualification argv/log')
        qlogs.append(pin(row['log']))
    checked_pin(qualification['prior_qualification'], 'preserved qualification01')
    require(Path(qualification['prior_qualification']['path']) == engineering / 'qualification-01.json', 'original qualification01 path')
    inputs.update(parent_admission=prior_inputs, parent_closure=plan['parent_closure'], parent_data=plan['data'],
                  parent_payloads=plan['parent_payloads'], qualification=plan['qualification'], qualification_logs=qlogs,
                  prior_qualification=qualification['prior_qualification'])
    manifest, paths = read(study / 'manifest.json'), list(study.rglob('*'))
    require(set(manifest) == {'files'} and not any(p.is_symlink() for p in paths), 'regular complete manifest')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(manifest['files']) | {'manifest.json', 'receipt.json'}, 'no omitted output files')
    for name, value in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and descriptor(study / name) == value, 'output file pin: ' + name)
    receipt = read(study / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == PLAN_SHA
            and [receipt[k] for k in ('fits', 'fresh_fit_attempts', 'cached_fit_records', 'rows', 'permutation_metric_rows',
                 'raw_decodes', 'reference_refits', 'parent_refits', 'saved_fit_loads', 'saved_dev_loads', 'confirmation_access', 'official_test_access')]
            == [48, 18, 30, 208, 36, 0, 0, 0, 7, 2, False, False]
            and 0 < receipt['seconds'] <= 14400 and receipt['clock'] == runtime['clock']
            and 0 < receipt['monotonic_seconds'] <= process['elapsed_seconds'], 'complete producer terminal')
    return plan, inputs

def scored(prediction, target, scale, horizon, np):
    if not np.isfinite(prediction).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        error = prediction[:, :horizon] - target[:, :horizon]
        sse = float(np.sum(error * error))
        joint = np.mean((error * scale) ** 2, axis=(0, 1))
    if not math.isfinite(sse) or not np.isfinite(joint).all():
        return None
    return {'standardized_rmse': math.sqrt(sse / error.size), 'standardized_sse': sse,
            'scalars': int(error.size), 'physical_rmse_deg': math.sqrt(float(np.mean(joint))),
            'per_joint_rmse_deg': np.sqrt(joint).tolist(), 'windows': len(target), 'horizon': horizon}


def decisions(rows, resources, cfg):
    """Independent pooled recipe choice and the five literal decision criteria."""
    dev = cfg['partitions']['dev']
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    expected = {(name, arm, seed, rate, h) for name in dev for arm in ARMS for seed in SEEDS for rate in RATES for h in (64, 128)}
    expected |= {(name, arm, None, None, h) for name in dev for arm in REFS for h in (64, 128)}
    require(len(rows) == len(lookup) == 208 and set(lookup) == expected, 'complete208 metric roster')
    def metric(name, arm, seed=None, rate=None):
        row = lookup[name, arm, seed, rate, 128]
        value = row['metrics']
        if row['status'] != 'PASS' or value is None:
            return None
        numbers = [value[k] for k in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')] + value['per_joint_rmse_deg']
        return value if (len(value['per_joint_rmse_deg']) == 6 and type(value['scalars']) is int and value['scalars'] > 0
                         and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in numbers)) else None
    options, chosen = {}, {}
    for arm in ARMS:
        eligible, options[arm] = [], []
        for rate in RATES:
            values = [metric(name, arm, seed, rate) for name in dev for seed in SEEDS]
            score = math.sqrt(sum(v['standardized_sse'] for v in values) / sum(v['scalars'] for v in values)) if all(v is not None for v in values) else None
            options[arm].append({'rate': rate, 'eligible': score is not None, 'pooled_rmse': score})
            if score is not None:
                eligible.append((score, rate))
        chosen[arm] = min(eligible)[1] if eligible else None
    selection = {'selected_rates': chosen, 'options': options,
                 'scope': 'pooled H128 exposed DEV2/all3seeds, lower-rate tie; cached full two-rate recipes replayed without fitting'}
    details, equal_means, costs = {}, {}, {}
    for name in dev:
        means, values_by_arm = {}, {}
        for arm in (*ARMS, *REFS):
            values = ({seed: metric(name, arm, seed, chosen[arm]) for seed in SEEDS} if chosen[arm] is not None else {}) if arm in ARMS else {None: metric(name, arm)}
            if values and all(value is not None for value in values.values()):
                values_by_arm[arm] = {seed: value['standardized_rmse'] for seed, value in values.items()}
                means[arm] = sum(values_by_arm[arm].values()) / len(values)
        paired = {other: {str(seed): values_by_arm['temporal_affine'][seed] - values_by_arm[other][seed] for seed in SEEDS}
                  for other in FRESH_ARMS[:2] if other in values_by_arm and 'temporal_affine' in values_by_arm}
        details[name] = {'means': means, 'paired_primary_differences': paired}
    for arm in (*ARMS, *REFS):
        if all(arm in details[name]['means'] for name in dev):
            equal_means[arm] = sum(details[name]['means'][arm] for name in dev) / len(dev)
        costs[arm] = None
        subset = [r for r in resources if r['arm'] == arm]
        seeds = set(SEEDS) if arm in ARMS else {None}
        if len(subset) != len(seeds) or {r['seed'] for r in subset} != seeds:
            continue
        if arm in ARMS and any(r.get('learning_rate') != chosen[arm] for r in subset):
            continue
        times = [r['timing']['median_seconds'] for r in subset]
        sizes = [[r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')] for r in subset]
        if (all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in times)
                and all(type(v) is int and v >= 0 for parts in sizes for v in parts)):
            costs[arm] = {'latency': sorted(times)[len(times)//2], 'bytes': max(sum(parts) for parts in sizes)}
    candidate = equal_means.get('temporal_affine')
    complete = all(chosen[a] is not None and a in equal_means for a in FRESH_ARMS)
    gain = candidate is not None and all(equal_means.get(a, 0) > 0 and candidate <= .95 * equal_means[a] for a in FRESH_ARMS[:2])
    harm = all(all(a in d['means'] for a in FRESH_ARMS)
               and d['means']['temporal_affine'] <= 1.02 * min(d['means'][a] for a in FRESH_ARMS[:2]) for d in details.values())
    left, right = costs['temporal_affine'], costs['last_two']
    latency = left is not None and right is not None and left['latency'] <= 1.25 * right['latency']
    frontier = all(a in equal_means and costs[a] is not None for a in (*ARMS, *REFS))
    dominators = []
    if frontier:
        point = (candidate, left['latency'], left['bytes'])
        for arm in (*ARMS, *REFS):
            if arm != 'temporal_affine':
                control = (equal_means[arm], costs[arm]['latency'], costs[arm]['bytes'])
                if all(a <= b for a, b in zip(control, point, strict=True)) and any(a < b for a, b in zip(control, point, strict=True)):
                    dominators.append(arm)
    conditions = [{'name': name, 'passed': bool(flag)} for name, flag in (
        ('primary_recipes_complete', complete), ('equal_file_mean_5pct_vs_both_locals', gain),
        ('each_file_within_2pct_best_local', harm), ('latency_within_125pct_last_two', latency),
        ('complete_frontier_not_dominated', frontier and not dominators))]
    passed = sum(row['passed'] for row in conditions)
    return selection, {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == 5 else 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION',
                       'passed': passed, 'total': 5, 'conditions': conditions, 'details': details,
                       'equal_file_means': equal_means, 'costs': costs, 'frontier_complete': frontier, 'dominators': dominators,
                       'scope': 'exposed-DEV information comparison; no novelty, independent confirmation or control claim'}


def validate_fit_evidence(initial, final, optimizer, parameter_shapes, receipt, np):
    """Pure saved-array shape/ownership/Adam ledger checks, no optimizer calls."""
    require(set(initial) == set(final) == set(parameter_shapes), 'checkpoint keys')
    for name, shape in parameter_shapes.items():
        require(initial[name].shape == final[name].shape == tuple(shape)
                and initial[name].dtype == final[name].dtype == np.float32 and np.isfinite(initial[name]).all(), 'checkpoint shape/dtype/initial finite')
    require(set(optimizer) <= {name + '/' + suffix for name in parameter_shapes for suffix in ('exp_avg', 'exp_avg_sq', 'step')}, 'Adam ownership')
    steps = receipt['completed_updates']
    for name, shape in parameter_shapes.items():
        keys = {name + '/' + suffix for suffix in ('exp_avg', 'exp_avg_sq', 'step')}
        require(not (keys & set(optimizer)) or keys <= set(optimizer), 'complete Adam slot triple')
        if keys <= set(optimizer):
            for suffix in ('exp_avg', 'exp_avg_sq'):
                require(optimizer[name + '/' + suffix].shape == tuple(shape)
                        and optimizer[name + '/' + suffix].dtype == np.float32, 'Adam moment shape/dtype')
            value = optimizer[name + '/step']
            require(value.shape == () and value.dtype == np.float32 and np.isfinite(value)
                    and float(value) in (steps, steps+1), 'Adam attempted update ledger')
    require(not optimizer or len(optimizer) == 3*len(parameter_shapes), 'all Adam parameters or unstarted')
    require(not steps or optimizer, 'completed steps require optimizer evidence')
    if receipt['status'] == 'PASS':
        require(steps == 4096 and len(optimizer) == 3*len(parameter_shapes)
                and all(np.isfinite(v).all() for v in (*final.values(), *optimizer.values())), 'finite complete successful evidence')
        require(all(float(optimizer[name + '/step']) == 4096 for name in parameter_shapes), 'successful4096 Adam steps')


def initial_pairing(initial, arm, np):
    cell = {name.removeprefix('cell.'): value for name, value in initial.items() if name.startswith('cell.')}
    require(sum(v.size for v in cell.values()) == 590, 'common590 parameters')
    heads = {name: value for name, value in initial.items() if not name.startswith('cell.')}
    require(set(heads) == (set() if arm == 'last_two' else {'head.weight', 'head.bias'}), 'exact head keys')
    if heads:
        require(heads['head.weight'].shape == (12, 30) and heads['head.bias'].shape == (12,)
                and all(np.count_nonzero(v) == 0 for v in heads.values()), 'zero372 initializer head')
    digest = hashlib.sha256()
    for name, value in cell.items():
        digest.update(name.encode() + b'\0' + str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes())
    return {'common_cell_sha256': digest.hexdigest(), 'zero_head': True, 'common_parameters': 590,
            'added_parameters': 0 if arm == 'last_two' else 372}


def validate_barrier(barrier, fits, checkpoint_pins):
    require(len(fits) == len(checkpoint_pins) == 48, 'all48 barrier entries')
    require(barrier == {'fresh_fit_attempts': 18, 'cached_fit_records': 30, 'fit_records': 48,
                       'dev_loads_this_run': 0, 'dev_exposed_prior': True, 'checkpoints': checkpoint_pins}, 'all18 fresh attempts precede DEV barrier')


def diagnostic_rows(common, original, prediction, target, scale, error, np):
    if common['learning_rate'] is None:
        require(original is prediction is error is None, 'unavailable recipe has no diagnostic inference')
        return {**common, 'status': 'UNAVAILABLE', 'reason': 'no complete selected primary recipe'}, metric_rows(
            common, None, target, scale, {'type': 'UnavailableSelectedRecipe'}, np)
    require(original is not None and np.isfinite(original).all(), 'finite ordinary forecast required')
    if prediction is not None:
        require(prediction.dtype == np.float64 and prediction.shape == original.shape == target.shape, 'diagnostic shape/dtype')
        if not np.isfinite(prediction).all():
            error = {'type': 'NonfiniteDiagnostic', 'message': 'nonfinite reversed-history prediction'}
    if error is not None:
        require(common['arm'] == 'temporal_affine' and error['type'] == 'NonfiniteDiagnostic', 'only temporal numerical diagnostic failure is retained')
        check = None
    else:
        require(prediction is not None, 'successful diagnostic needs prediction')
        exact = bool(np.array_equal(original, prediction))
        require(common['arm'] == 'temporal_affine' or exact, 'local exact permutation invariance')
        check = {'exact_invariance': exact, 'maximum_absolute_change': float(np.max(np.abs(original-prediction))),
                 'scope': 'fixed older-order corruption diagnostic only; not a sixth scientific gate'}
    return {**common, 'status': 'FAILED' if error else 'PASS', 'error': error, 'check': check,
            'prediction_saved': prediction is not None}, metric_rows(common, prediction, target, scale, error, np)

def validate_fit_receipt(receipt, trace, rate, cap):
    require(set(receipt) == {'status', 'error', 'completed_updates', 'requested_updates', 'learning_rate',
                            'optimizer_seconds', 'fit_seconds', 'fit_cap_scope', 'timing_scope', 'files'}, 'fit receipt schema')
    require(receipt['status'] in ('PASS', 'FAILED') and type(receipt['completed_updates']) is int
            and len(trace) == receipt['completed_updates'] <= 4096 and receipt['requested_updates'] == 4096
            and receipt['learning_rate'] == rate, 'update ledger')
    for update, row in enumerate(trace, 1):
        require(set(row) == {'update', 'loss', 'gradient_norm_before_clip'} and type(row['update']) is int
                and row['update'] == update and all(type(row[k]) in (int, float) and math.isfinite(row[k])
                and row[k] >= 0 for k in ('loss', 'gradient_norm_before_clip')), 'finite completed trace')
    require(all(type(receipt[k]) in (int, float) and math.isfinite(receipt[k]) for k in ('optimizer_seconds', 'fit_seconds'))
            and 0 <= receipt['optimizer_seconds'] <= receipt['fit_seconds'] and receipt['fit_seconds'] > 0, 'timing scopes')
    require(receipt['fit_cap_scope'] == 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows'
            and receipt['timing_scope'] == 'optimizer loop including batch construction and finite checks, excluding model construction and saved files', 'exact fit timing scopes')
    require(set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json'}, 'fit evidence files')
    if receipt['status'] == 'PASS':
        require(len(trace) == 4096 and receipt['error'] is None and receipt['fit_seconds'] <= cap, 'complete successful fit')
    else:
        require(isinstance(receipt['error'], dict) and set(receipt['error']) == {'type', 'message'}, 'retained failure reason')
        require(receipt['error']['type'] == 'FitFailure' and receipt['error']['message'] in (
            'single-fit wall cap exceeded', 'single-fit wall cap exceeded during preservation',
            'native single-fit deadline exceeded', 'nonfinite autoregressive training loss', 'nonfinite training gradient norm',
            'nonfinite updated parameters', 'nonfinite updated Adam state',
            'nonfinite structured transition output; no repair',
            'finite CPU tensor with exact shape/dtype: initializer features'), 'qualified finite failure type')

def metric_rows(common, prediction, target, scale, error, np):
    result = []
    for horizon in (64, 128):
        local, value = error, None
        if local is None:
            require(prediction.dtype == np.float64 and prediction.shape == target.shape, 'full forecast schema')
            value = scored(prediction, target, scale, horizon, np)
            if value is None:
                local = {'type': 'NonfiniteEvaluation', 'message': 'nonfinite prediction/target' if not np.isfinite(prediction).all() else 'nonfinite metric arithmetic'}
        result.append({**common, 'horizon': horizon, 'status': 'PASS' if local is None else 'FAILED', 'error': local, 'metrics': value})
    return result

def reference_prediction(arm, coefficients, batch, np):
    q, u, future = (batch[key] for key in ('q_context', 'u_context', 'future_u'))
    if arm in REFS[:2]:
        design = np.concatenate((q[:, 16:].reshape(len(q), 96), u[:, 15:31].reshape(len(q), 96),
                                 future.reshape(len(q), 768), np.ones((len(q), 1))), axis=1)
        result = np.empty((len(q), 128, 6), np.float64)
        for h in range(1, 129):
            indices = np.r_[np.arange(192+6*h), 960]
            result[:, h-1] = design[:, indices] @ coefficients[arm][f'h{h:03d}'].T
        return result
    if arm == 'persistence':
        return np.repeat(q[:, -1:, :], 128, axis=1)
    current, previous, previous_u = q[:, -1].copy(), q[:, -2].copy(), u[:, -2].copy()
    outputs = []
    with np.errstate(over='ignore', invalid='ignore'):
        for torque in future.transpose(1, 0, 2):
            feature = np.concatenate((current, previous, torque, previous_u, np.ones((len(q), 1))), axis=1)
            prediction = feature @ coefficients['linear_frozen'].T
            outputs.append(prediction)
            previous, current, previous_u = current, prediction, torque
    return np.stack(outputs, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--run-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not (args.output.parent / 'manifest.json').exists(), 'fresh audit output only')
    require(not args.output.resolve().is_relative_to(args.study.resolve()), 'audit must not mutate original study')
    result = audit(args.study, args.run_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    with (args.output.parent / 'manifest.json').open('x') as handle:
        json.dump({'files': {args.output.name: descriptor(args.output)}}, handle, sort_keys=True, indent=2)
        handle.write('\n')


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_history_initialization_study as replay
    import torch
    torch.set_num_threads(1)
    cfg = plan['config']
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'fit_attempts': 48, 'fresh_fit_attempts': 18, 'cached_fit_records': 30, 'checkpoint_replays': 0,
              'prediction_files': 0, 'metric_rows': 208, 'diagnostic_metric_rows': 36, 'diagnostic_replays': 0, 'diagnostic_prediction_files': 0, 'saved_batches': 3, 'dev_target_windows': 0,
              'raw_mat_decodes': 0, 'confirmation_decodes': 0, 'official_test_decodes': 0, 'optimizer_updates': 0}
    expected = {'registration.json', 'runtime.json', 'normalizers.npz', 'linear.npz',
                'causal_ridge_1.npz', 'causal_ridge_100.npz', 'causal_ridge_1.json', 'causal_ridge_100.json',
                'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json', 'permutation.json',
                'parent-structured-fits.json', 'parent-structured-results.json', 'parent-transition-fits.json', 'parent-transition-results.json'} | {'sources/' + name for name in SOURCES}
    def load(name, parent=False):
        if not parent:
            expected.add(name)
        path = plan['data'][name]['path'] if parent else study / name
        with np.load(path, allow_pickle=False) as data:
            value = {key: data[key].copy(order='K') for key in data.files}
        counts['npz_decodes'] += 1
        counts['array_decodes'] += len(value)
        return value
    def equal(a, b, label):
        require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True), label)
    norm = load('normalizers.npz')
    require(set(norm) == {'q_mean', 'q_std', 'u_mean', 'u_std'}, 'normalizer fields')
    require(all(value.dtype == np.float64 and value.shape == (6,) and np.isfinite(value).all()
                for value in norm.values()), 'finite float64 normalizer vectors')
    ancestor_norm = load('normalizers.npz', parent=True)
    require(set(ancestor_norm) == set(norm), 'all normalizer fields')
    for name in norm:
        equal(norm[name], ancestor_norm[name], 'same original normalizer')
    records = {}
    for partition in ('fit', 'dev'):
        records[partition] = []
        for name in cfg['partitions'][partition]:
            data = load(f'{partition}-data-{name}.npz', parent=True)
            require(set(data) == {'q', 'u', 'raw_indices'}, 'saved recording fields')
            for key in ('q', 'u'):
                require(data[key].dtype == np.float64 and data[key].shape == (3636, 6) and np.isfinite(data[key]).all(), 'finite saved physical recording')
            equal(data['raw_indices'], np.arange(0, 90881, 25, dtype=np.int64), 'causal sampling indices')
            records[partition].append({'name': name, **data})
    for kind in ('q', 'u'):
        combined = np.concatenate([r[kind][64:] for r in records['fit']])
        equal(norm[kind + '_mean'], combined.mean(0), 'FIT mean')
        equal(norm[kind + '_std'], combined.std(0), 'FIT population scale')
        require(np.all(norm[kind + '_std'] > 0), 'positive scales')
    lengths = [len(r['q']) for r in records['fit']]
    for seed in SEEDS:
        batch = load(f'batches-{seed}.npz')
        require(set(batch) == {'record', 'start'}, 'batch fields')
        rng = np.random.Generator(np.random.PCG64(seed + 520000))
        ids = rng.integers(0, 7, size=(4096, 16), dtype=np.int64)
        starts = np.empty_like(ids)
        for index in np.ndindex(ids.shape):
            starts[index] = rng.integers(64, lengths[int(ids[index])] - 32 - 128 + 1)
        equal(batch['record'], ids, 'paired recording order')
        equal(batch['start'], starts, 'paired window starts')
    dev = {}
    for record in records['dev']:
        name = record['name']
        q = (record['q'] - norm['q_mean']) / norm['q_std']
        u = (record['u'] - norm['u_mean']) / norm['u_std']
        starts = np.arange(64, len(q) - 32 - 128 + 1, 160, dtype=np.int64)
        batch = {'q_context': np.stack([q[s:s+32] for s in starts]),
                 'u_context': np.stack([u[s:s+32] for s in starts]),
                 'future_u': np.stack([u[s+31:s+159] for s in starts])}
        target = np.stack([q[s+32:s+160] for s in starts])
        saved = load('dev-windows-' + name + '.npz')
        require(set(saved) == {'starts', 'target'}, 'DEV window fields')
        equal(saved['starts'], starts, 'DEV starts')
        equal(saved['target'], target, 'independent DEV target alignment')
        dev[name] = (batch, target)
        counts['dev_target_windows'] += len(starts)
    linear_file = load('linear.npz')
    require(set(linear_file) == {'coefficient'}, 'linear schema')
    linear = linear_file['coefficient']
    require(linear.dtype == np.float64 and linear.shape == (6, 25) and np.isfinite(linear).all(), 'linear coefficients')
    parent_coefficients = load('references.npz', parent=True)
    equal(linear, parent_coefficients['linear_frozen'], 'unchanged FIT-only linear initializer')
    coefficients = {'linear_frozen': linear}
    for arm in REFS[:2]:
        bank = load(arm + '.npz')
        require(set(bank) == {f'h{h:03d}' for h in range(1, 129)}, 'complete causal ridge bank')
        for h in range(1, 129):
            value = bank[f'h{h:03d}']
            require(value.shape == (6, 193+6*h) and value.dtype == np.float64 and np.isfinite(value).all(), 'causal coefficient shape')
        metadata = read(study / (arm + '.json'))
        require(metadata['coefficient_count'] == 445440 and metadata['coefficient_bytes'] == 3563520
                and metadata['retained_numeric_bytes'] == 3563536 and metadata['intercept_penalized'] is True
                and metadata['penalty'] == (1. if arm.endswith('_1') else 100.)
                and metadata['fit_rows'] == sum(len(np.arange(64, n-32-128+1, 32)) for n in lengths), 'causal bank metadata')
        require(math.isfinite(metadata['fit_seconds']) and metadata['fit_seconds'] >= 0, 'ridge time attestation')
        coefficients[arm] = bank

    fits = read(study / 'fits.json')
    parents = {group: read(study / f'parent-{group}-fits.json') for group in ('structured', 'transition')}
    originals, roster = {}, [(a, s, r, f'{a}-{s}-lr{i}', None, None)
                             for s in SEEDS for i, r in enumerate(RATES) for a in FRESH_ARMS]
    for group, ledger in parents.items():
        wanted = CACHED_ARMS[:-1] if group == 'structured' else ('gru_residual',)
        require(len(ledger) == (36 if group == 'structured' else 24), 'full original parent ledger')
        subset = [row for row in ledger if row['arm'] in wanted]
        expected_keys = {f'{a}-{s}-lr{i}' for a in wanted for s in SEEDS for i in range(2)}
        require(len(subset) == len(expected_keys) and {f['key'] for f in subset} == expected_keys, 'all cached rates/seeds')
        for row in subset:
            arm = 'gru10' if group == 'transition' else row['arm']
            require(row['seed'] in SEEDS and row['learning_rate'] in RATES
                    and row['key'] == f"{row['arm']}-{row['seed']}-lr{RATES.index(row['learning_rate'])}", 'original fit identity')
            key = row['key'].replace('gru_residual-', 'gru10-', 1) if group == 'transition' else row['key']
            originals[(group, row['key'])] = row
            roster.append((arm, row['seed'], row['learning_rate'], key, group, row['key']))
    require(len(fits) == len(roster) == 48 and len({r[3] for r in roster}) == 48, 'all48 ordered attempts')
    models, initials, common_hashes, failed = {}, {}, {}, 0
    for number, (fit, (arm, seed, rate, key, group, parent_key)) in enumerate(zip(fits, roster, strict=True), 1):
        require((fit['arm'], fit['seed'], fit['learning_rate'], fit['key']) == (arm, seed, rate, key), 'attempt order')
        fresh = arm in FRESH_ARMS
        fields = {'key', 'arm', 'seed', 'learning_rate', 'fit', 'resources', 'origin', 'effective_status'}
        if fresh:
            fields |= {'native_fit_seconds', 'native_fit_error', 'native_fit_cap_seconds', 'initial_pairing'}
            require(fit['origin'] == 'fresh' and fit['native_fit_cap_seconds'] == 1800.
                    and type(fit['native_fit_seconds']) in (int, float) and math.isfinite(fit['native_fit_seconds'])
                    and fit['native_fit_seconds'] > 0, 'fresh native duration')
            late = fit['native_fit_seconds'] >= 1800.
            require(fit['effective_status'] == ('FAILED' if late else fit['fit']['status'])
                    and fit['native_fit_error'] == ({'type': 'FitFailure', 'message': 'native fit cap crossed including preservation'} if late else None), 'preserved native terminal override')
        else:
            fields |= {'parent_group', 'parent_key', 'parent_fit', 'parent_files'}
            original = originals[group, parent_key]
            require(fit['origin'] == 'cached_parent' and fit['parent_group'] == group and fit['parent_key'] == parent_key
                    and fit['parent_fit'] == original and fit['fit'] == original['fit'] and fit['resources'] == original['resources']
                    and fit['effective_status'] == original.get('effective_status', original['fit']['status'])
                    and fit['parent_files'] == {name: plan['parent_payloads'][group][parent_key + '/' + name] for name in LEGACY_FILES}, 'original cached lineage/status')
        require(set(fit) == fields, 'fit record fields')
        expected.update({f'completed-fit-{number:02d}.json', key + '/fit-receipt.json', key + '/trace.json'})
        require(read(study / f'completed-fit-{number:02d}.json') == fit, 'immediate fit record')
        receipt, trace = read(study / key / 'fit-receipt.json'), read(study / key / 'trace.json')
        require(receipt == fit['fit'], 'fit receipt exact original')
        validate_fit_receipt(receipt, trace, rate, 1800 if fresh else 600 if arm in ('legacy_instant', 'gru10') else 900)
        for name, value in receipt['files'].items():
            require(descriptor(study / key / name) == value, 'fit evidence pin')
        model = replay.model_for(arm, seed, linear)
        initial, final, optimizer = (load(key + '/' + suffix + '.npz') for suffix in ('initial', 'final', 'optimizer'))
        initials[key] = initial
        template = model.state_dict()
        require(set(initial) == set(template), 'factory initial parameter keys')
        for name, tensor in template.items():
            equal(initial[name], tensor.detach().numpy(), 'exact seeded initial factory')
        parameter_shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
        validate_fit_evidence(initial, final, optimizer, parameter_shapes, receipt, np)
        count = sum(p.numel() for p in model.parameters())
        state_count = 50 if arm == 'gru32' else 28 if arm == 'gru10' else 12
        require(count == PARAMETERS[arm] and not list(model.buffers()), 'qualified parameter/buffer roster')
        resource = fit['resources']
        require(set(resource) == {'parameters', 'parameter_bytes', 'buffer_bytes', 'state_scalars', 'state_bytes',
                                  'normalizer_bytes', 'inactive_parameters', 'dtype', 'input_bytes', 'output_bytes', 'temporary_workspace'}, 'fit resource schema')
        require([resource[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, 4*count, state_count, 4*state_count, 0, 192]
                and resource['inactive_parameters'] == (192 if arm == 'legacy_instant' else 0)
                and resource['dtype'] == 'float32' and resource['input_bytes'] == 9216
                and resource['output_bytes'] == 6144 and resource['temporary_workspace'] == 'not measured', 'complete all-fit resources')
        if fresh:
            pairing = initial_pairing(initial, arm, np)
            require(pairing == fit['initial_pairing'], 'saved independent initializer hash')
            common_hashes.setdefault(seed, pairing['common_cell_sha256'])
            require(common_hashes[seed] == pairing['common_cell_sha256'], 'all paired590 initial weights identical')
        if fit['effective_status'] == 'PASS':
            model.load_state_dict({name: torch.from_numpy(value) for name, value in final.items()}, strict=True)
            model.eval()
            require(all(p.grad is None for p in model.parameters()), 'no replay parameter gradients')
            models[key] = model
        else:
            failed += 1
    counts['initial_pairing_checks'] = 18
    validate_barrier(read(study / 'checkpoint-barrier.json'), fits,
                     {f['key']: descriptor(study / f['key'] / 'final.npz') for f in fits})
    saved_result = read(study / 'results.json')
    require(saved_result['version'] == cfg['version'] and saved_result['config'] == cfg, 'result identity')
    rows, ordinary = [], {}
    numerical_errors = ('nonfinite LPV output; no rollout clipping or repair', 'nonfinite structured transition output; no repair',
                        'finite CPU tensor with exact shape/dtype: initializer features')
    for name, (batch, target) in dev.items():
        for fit in fits:
            key = fit['key']
            common = {k: fit[k] for k in ('arm', 'seed', 'learning_rate')}
            common.update(recording=name, fit_key=key)
            error, prediction = None, None
            filename = 'prediction-' + name + '-' + key + '.npz'
            if key not in models:
                require(not (study / filename).exists(), 'failed fit cannot emit forecast')
                error = {'type': 'FailedTrainingAttempt', 'effective_status': fit['effective_status']}
            else:
                counts['checkpoint_replays'] += 1
                try:
                    with torch.no_grad():
                        replayed = replay.old.infer(models[key], batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) in numerical_errors and not (study / filename).exists(), 'known forecast failure without bank')
                    error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                else:
                    saved = load(filename)
                    require(set(saved) == {'prediction'}, 'forecast bank field')
                    prediction = saved['prediction']
                    equal(prediction, replayed, 'exact same-runtime qualified checkpoint replay')
                    counts['prediction_files'] += 1
                    if fit['arm'] in FRESH_ARMS:
                        ordinary[name, key] = prediction
            rows.extend(metric_rows(common, prediction, target, norm['q_std'], error, np))
        for arm in REFS:
            saved = load('prediction-' + name + '-' + arm + '.npz')
            require(set(saved) == {'prediction'}, 'reference bank field')
            prediction = saved['prediction']
            equal(prediction, reference_prediction(arm, coefficients, batch, np), 'independent reference forecast')
            counts['prediction_files'] += 1
            counts['reference_replays'] = counts.get('reference_replays', 0) + 1
            rows.extend(metric_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None},
                                    prediction, target, norm['q_std'], None, np))
    close(rows, saved_result['rows'], 'all208 independent ordinary scores')
    resources = read(study / 'resources.json')
    selection, result = decisions(rows, resources, cfg)
    close(selection, saved_result['selection'], 'independent rates')
    close(result, saved_result['result'], 'five independent science conditions')
    require(read(study / 'receipt.json')['scientific_result'] == result['status'], 'terminal scientific outcome')
    validate_resources(resources, fits, selection, rows, coefficients, np)
    diagnostic_checks, diagnostic_metrics = [], []
    for name, (batch, target) in dev.items():
        changed = {**batch, 'q_context': batch['q_context'][:, list(PERMUTATION), :].copy(),
                   'u_context': batch['u_context'][:, list(PERMUTATION), :].copy()}
        for arm in FRESH_ARMS:
            rate = selection['selected_rates'][arm]
            for seed in SEEDS:
                common = {'recording': name, 'arm': arm, 'seed': seed, 'learning_rate': rate}
                original, prediction, error = None, None, None
                if rate is not None:
                    key = f'{arm}-{seed}-lr{RATES.index(rate)}'
                    original = ordinary[name, key]
                    filename = 'permuted-prediction-' + name + '-' + key + '.npz'
                    counts['diagnostic_replays'] += 1
                    try:
                        with torch.no_grad():
                            replayed = replay.old.infer(models[key], changed).numpy().astype(np.float64)
                    except replay.old.FitFailure as exc:
                        require(arm == 'temporal_affine' and str(exc) in numerical_errors
                                and not (study / filename).exists(), 'only retained temporal numerical OOD failure')
                        error = {'type': 'NonfiniteDiagnostic', 'message': str(exc)}
                    else:
                        saved = load(filename)
                        require(set(saved) == {'prediction'}, 'diagnostic bank field')
                        prediction = saved['prediction']
                        equal(prediction, replayed, 'exact qualified reversal replay')
                        counts['diagnostic_prediction_files'] += 1
                check, metrics = diagnostic_rows(common, original, prediction, target, norm['q_std'], error, np)
                diagnostic_checks.append(check)
                diagnostic_metrics.extend(metrics)
    permutation = {'permutation': list(PERMUTATION), 'checks': diagnostic_checks, 'rows': diagnostic_metrics,
                   'scientific_gate': False,
                   'scope': 'paired older-order corruption; local exact invariance is a diagnostic validity condition'}
    close(permutation, read(study / 'permutation.json'), 'all36 nongating diagnostic rows and18 attempts')
    require(len(diagnostic_checks) == 18 and len(diagnostic_metrics) == 36, 'complete selected-primary diagnostic roster')
    require(set(read(study / 'manifest.json')['files']) == expected, 'complete exact dynamic output inventory')
    counts.update(failed_fit_attempts=failed, passed_fit_attempts=48-failed,
                  failed_metric_rows=sum(r['status'] == 'FAILED' for r in rows), resource_rows=len(resources),
                  manifest_files=len(expected), condition_rows=5, initial_checkpoints=48, final_checkpoints=48, optimizer_checkpoints=48,
                  fresh_requested_training_updates=73728, historical_requested_training_updates=122880,
                  completed_training_updates=sum(f['fit']['completed_updates'] for f in fits if f['origin'] == 'fresh'),
                  archived_training_updates=sum(f['fit']['completed_updates'] for f in fits if f['origin'] == 'cached_parent'),
                  failed_diagnostic_attempts=sum(c['status'] == 'FAILED' for c in diagnostic_checks),
                  unavailable_diagnostic_attempts=sum(c['status'] == 'UNAVAILABLE' for c in diagnostic_checks))
    require(authenticate(study, run_receipt) == (plan, inputs), 'unchanged original evidence after replay')
    return {'version': 'robot-history-initialization-audit-v1', 'status': 'PASS', 'agreement': True, 'inputs': inputs,
            'auditor': pin(__file__), 'study': str(study), 'registration_sha256': PLAN_SHA, 'source_pins': plan['sources'],
            'results': {'rows': rows, 'selection': selection, 'result': result}, 'permutation': permutation,
            'resources': resources, 'counts': counts, 'seconds': time.monotonic()-started,
            'checks': {'complete_original_closure': True, 'independent_fit_normalization_and_batches': True,
                       'independent_dev_targets': True, 'independent_metrics_selection_rules': True,
                       'exact_qualified_checkpoint_replay': True, 'all_failed_attempts_retained': True,
                       'inherited_controls_byte_identical': True, 'all48_checkpoint_barrier': True,
                       'common_initial590_and_zero_heads': True, 'permutation_separate_from_science_rule': True},
            'scope': ['No raw MAT, CONFIRM/TEST, new fitting or optimizer updates.',
                      'All ordinary/permuted scores, rate selection, five rules and causal reference forecasts are independent arithmetic.',
                      'Checkpoint inference reuses qualified frozen implementations, not an independent recurrence.',
                      'Filtering provenance is authenticated without re-filtering; training gradients and timing are not rerun.',
                      'All18 fresh and30 historical attempts remain visible. Negative outcomes can have audit agreement.']}


def validate_resources(resources, fits, selection, rows, coefficients, np):
    selected = [f for f in fits if selection['selected_rates'][f['arm']] == f['learning_rate']]
    references = [arm for arm in REFS if len([r for r in rows if r['arm'] == arm and r['horizon'] == 128]) == 2
                  and all(r['status'] == 'PASS' and r['metrics'] is not None for r in rows if r['arm'] == arm and r['horizon'] == 128)]
    require(len(resources) == len(selected) + len(references), 'complete available timing roster')
    for row, fit in zip(resources[:len(selected)], selected, strict=True):
        require((row['arm'], row['seed'], row['learning_rate']) == (fit['arm'], fit['seed'], fit['learning_rate']), 'selected resource identity')
        close({k: row[k] for k in fit['resources']}, fit['resources'], 'fit numeric storage join')
    require([r['arm'] for r in resources[len(selected):]] == references, 'available reference timing order')
    for row in resources:
        arm, neural = row['arm'], row['arm'] in ARMS
        count = PARAMETERS[arm] if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
        state = (50 if arm == 'gru32' else 28 if arm == 'gru10' else 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
        require([row[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, count*(4 if neural else 8), state, state*(4 if neural else 8), 16 if arm in REFS[:2] else 0, 192], 'complete retained numeric storage')
        require(row['dtype'] == ('float32' if neural else 'float64') and row['input_bytes'] == 9216
                and row['output_bytes'] == 6144 and row['temporary_workspace'] == 'not measured', 'request/precision scope')
        if neural:
            require(row['inactive_parameters'] == (192 if arm == 'legacy_instant' else 0), 'inactive parameters explicit')
        timing = row['timing']
        require(set(timing) == {'seconds', 'median_seconds', 'p95_seconds', 'scope'}
                and timing['scope'] == 'batch1 normalization, casting, context32, future128, denormalization and finite check; no model/disk load; outer validation and operator preparation included; eager CPU', 'complete request timing scope')
        samples = timing['seconds']
        require(len(samples) == 20 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples), 'twenty positive finite timings')
        close(float(np.median(samples)), timing['median_seconds'], 'timing median')
        close(float(np.percentile(samples, 95)), timing['p95_seconds'], 'timing p95')


if __name__ == '__main__':
    main()
