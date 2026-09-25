"""Closed-output audit; scalar rules independent, checkpoint inference qualified/reused."""
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
PLAN_SHA = 'e1b48ab32cac7d72b85fd204034d73c191060cb98ced1d86e046a1de03d04782'
PARENT_SHA = '0dc6ff429e2b320844b9051f2756d2d1890039a05c7e840ac444f4f453f05880'
COMMIT = '24b124bb9dac330729b79c277933ff50ac18ba66'
FRESH_ARMS = ('householder12',)
CACHED_ARMS = ('householder', 'dense_bounded', 'dense_unbounded', 'dense_mlp', 'gru32', 'legacy_instant')
ARMS = (*FRESH_ARMS, *CACHED_ARMS)
PARAMETERS = dict(zip(ARMS, (806, 630, 806, 806, 590, 5916, 1014), strict=True))
LEGACY_FILES = ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')
PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
            'causal_ridge_100.npz', 'causal_ridge_100.json', 'fits.json',
            *(f'batches-{seed}.npz' for seed in (8101, 8102, 8103)),
            *(f'{arm}-{seed}-lr{ri}/{name}' for arm in CACHED_ARMS for seed in (8101, 8102, 8103)
              for ri in range(2) for name in LEGACY_FILES))
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES = (8101, 8102, 8103), (.001, .003)
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py',
           'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py',
           'tests/test_industrial_robot_data.py', 'tests/test_robot_coupling_study.py',
           'research/robot-coupling-protocol.md', 'src/openjev/research/bounded_robot_transition.py',
           'src/openjev/research/causal_robot_ridge.py', 'scripts/robot_transition_study.py',
           'tests/test_bounded_robot_transition.py', 'tests/test_causal_robot_ridge.py',
           'tests/test_robot_transition_study.py', 'research/robot-transition-protocol.md',
           'src/openjev/research/compact_robot_gate.py', 'tests/test_compact_robot_gate.py',
           'src/openjev/research/structured_robot_transition.py', 'tests/test_structured_robot_transition.py',
           'scripts/robot_structured_study.py', 'tests/test_robot_structured_study.py',
           'research/robot-structured-protocol.md', 'src/openjev/research/reflection_capacity.py',
           'tests/test_reflection_capacity.py', 'scripts/robot_reflection_capacity_study.py',
           'tests/test_robot_reflection_capacity_study.py', 'research/robot-reflection-capacity-protocol.md',
           'scripts/plot_robot_structured.py', 'scripts/audit_robot_structured.py',
           'src/openjev/research/suspend_clock.py')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_reflection_capacity_study.py', '--registration',
           'research/robot-reflection-capacity-registration.json', '--output', 'output/robot-reflection-capacity-study-v1']


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


def authenticate(study, run_receipt):
    """Metadata/opaque bytes only; no numerical import or checkpoint decoding."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    require(PLAN_SHA != 'UNBOUND' and COMMIT != 'UNBOUND', 'bind original registration and prefit commit first')
    require(study == ROOT / 'output/robot-reflection-capacity-study-v1', 'fixed original study')
    engineering = ROOT / 'output/robot-reflection-capacity-engineering-v1'
    require(run_receipt == engineering / 'run-process-01.json', 'original process path')
    registration = ROOT / 'research/robot-reflection-capacity-registration.json'
    require(descriptor(registration)['sha256'] == PLAN_SHA, 'frozen registration')
    plan = read(registration)
    require(plan['version'] == 'robot-reflection-capacity-study-v1' and set(plan['sources']) == set(SOURCES), 'source roster')
    for name, value in plan['sources'].items():
        require(descriptor(ROOT / name) == value == descriptor(study / 'sources' / name), 'source/snapshot: ' + name)
    require(descriptor(study / 'registration.json') == descriptor(registration), 'byte-exact saved registration')
    import audit_robot_structured as parent_admission
    parent_folder = ROOT / 'output/robot-structured-study-v1'
    parent_engineering = ROOT / 'output/robot-structured-engineering-v1'
    prior, prior_inputs = parent_admission.authenticate(parent_folder, parent_engineering / 'run-process-01.json')
    parent_registration = ROOT / 'research/robot-structured-registration.json'
    require(descriptor(parent_registration)['sha256'] == PARENT_SHA == plan['parent_registration_sha256'], 'closed parent registration')
    require(set(prior['sources']) == set(SOURCES[:21]) and all(prior['sources'][n] == plan['sources'][n] for n in SOURCES[:21]), 'unchanged parent sources')
    cfg = dict(prior['config'])
    cfg.update(version='robot-reflection-capacity-study-v1', arms=list(FRESH_ARMS), comparison_arms=list(ARMS),
               fit_cap_seconds=1800., wall_cap_seconds=10800., latency_ratio=1.05, r4_mean_ratio=.95,
               r4_paired_ratio=1., cached_parent_refit=False, reflections=12, parameters=806, state_scalars=12)
    del cfg['storage_ratio']
    require(plan['config'] == cfg, 'unchanged inherited and exact prospective config')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread environment')
    runtime = read(study / 'runtime.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform()
            and runtime['machine'] == platform.machine() and runtime['torch_threads'] == 1
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1'), 'same current runtime/platform')
    for package in ('numpy', 'torch'):
        require(importlib.metadata.version(package) == runtime[package], 'same package: ' + package)
    require(runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'), 'native suspend clock')
    launcher = ROOT / 'scripts/launch_robot_reflection_capacity.py'
    require(descriptor(launcher) == plan['launcher'], 'registered launcher')
    process, launch = read(run_receipt), read(engineering / 'run-launch-01.json')
    launch_keys = {'command', 'prefit_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == launch_keys and set(process) == launch_keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'}, 'original launch/process schema')
    require({k: process[k] for k in launch_keys} == launch and launch['command'] == COMMAND
            and launch['prefit_commit'] == COMMIT and launch['registration_sha256'] == PLAN_SHA
            and launch['launcher'] == plan['launcher'] and launch['thread_env'] == dict.fromkeys(THREADS, '1'), 'original launch joins')
    require(type(process['returncode']) is int and process['returncode'] == 0 and process['external_timeout'] is False
            and math.isfinite(process['elapsed_seconds']) and 0 < process['elapsed_seconds'] <= cfg['wall_cap_seconds'] + 60
            and descriptor(run_receipt.with_suffix('.log')) == process['log'], 'successful original process and log')
    inputs = {key: pin(path) for key, path in {
        'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'), 'run_launch': engineering / 'run-launch-01.json',
        'registration': registration, 'launcher': launcher, 'parent_registration': parent_registration,
        'manifest': study / 'manifest.json', 'producer_receipt': study / 'receipt.json', 'runtime': study / 'runtime.json'}.items()}
    closure = plan['parent_closure']
    require(set(closure) == {'manifest', 'receipt', 'process', 'audit', 'audit_process'}, 'closed parent roster')
    parent = {key: read(checked_pin(item, 'parent/' + key)['path']) for key, item in closure.items()}
    canonical = {'manifest': parent_folder / 'manifest.json', 'receipt': parent_folder / 'receipt.json',
                 'process': parent_engineering / 'run-process-01.json',
                 'audit': ROOT / 'output/robot-structured-audit-v1/audit.json',
                 'audit_process': parent_engineering / 'audit-process-01.json'}
    require(all(Path(closure[k]['path']) == path for k,path in canonical.items()), 'original parent paths')
    require(parent['receipt']['status'] == 'PASS' and parent['process']['returncode'] == 0
            and parent['audit_process']['returncode'] == 0 and parent['audit']['status'] == 'PASS'
            and parent['audit']['agreement'] is True and parent['receipt']['registration_sha256'] == PARENT_SHA
            and parent['process']['registration_sha256'] == PARENT_SHA
            and parent['audit']['registration_sha256'] == PARENT_SHA, 'closed parent')
    require(parent['audit']['inputs'] == prior_inputs and parent['audit']['source_pins'] == prior['sources']
            and parent['audit']['auditor'] == pin(ROOT / 'scripts/audit_robot_structured.py')
            and parent['audit_process']['audit_output'] == {k: closure['audit'][k] for k in ('sha256', 'bytes')}, 'parent audit terminal join')
    expected_data = {f'{split}-data-{name}.npz' for split in ('fit', 'dev') for name in cfg['partitions'][split]}
    expected_data |= {'normalizers.npz', 'references.npz'}
    require(set(plan['data']) == expected_data and plan['data'] == prior['data'] == parent['audit']['inputs']['parent_data'], 'exact saved data roster')
    for name, item in plan['data'].items():
        checked_pin(item, 'saved data/' + name)
    require(set(plan['parent_payloads']) == set(PAYLOADS) and len(PAYLOADS) == 190, '190 cached parent payloads')
    for name, item in plan['parent_payloads'].items():
        checked_pin(item, 'parent payload/' + name)
        require(Path(item['path']) == parent_folder / name
                and {k: item[k] for k in ('bytes', 'sha256')} == parent['manifest']['files'][name], 'parent manifest join: ' + name)
        copied = 'parent-normalizers.npz' if name == 'normalizers.npz' else 'parent-fits.json' if name == 'fits.json' else name
        require(descriptor(study / copied) == {k: item[k] for k in ('bytes', 'sha256')}, 'byte-identical cached evidence: ' + name)
    qual = checked_pin(plan['qualification'], 'qualification')
    qualification = read(qual['path'])
    require(Path(qual['path']) == engineering / 'qualification-01.json' and qualification['status'] == 'PASS'
            and qualification['sources'] == plan['sources'] and qualification['launcher'] == plan['launcher']
            and qualification['sources_unchanged'] is True and qualification['thread_env'] == dict.fromkeys(THREADS, '1')
            and len(qualification['commands']) == 2, 'qualified exact sources')
    require(datetime.fromisoformat(qualification['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification and registration precede original launch')
    commands = [['.venv/bin/ruff', 'check', 'src/openjev/research/reflection_capacity.py', 'tests/test_reflection_capacity.py',
                 'scripts/robot_reflection_capacity_study.py', 'tests/test_robot_reflection_capacity_study.py', 'scripts/launch_robot_reflection_capacity.py'],
                ['.venv/bin/python', '-m', 'pytest', '-q', 'tests/test_reflection_capacity.py', 'tests/test_robot_reflection_capacity_study.py']]
    qlogs = []
    for i, (row, command) in enumerate(zip(qualification['commands'], commands, strict=True), 1):
        require(row['command'] == command and row['returncode'] == 0
                and Path(row['log']) == engineering / f'qualification-01-command-{i}.log'
                and math.isfinite(row['seconds']) and row['seconds'] > 0
                and descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification logs/argv')
        qlogs.append(pin(row['log']))
    inputs.update(parent_closure=closure, parent_data=plan['data'], parent_payloads=plan['parent_payloads'],
                  parent_admission=prior_inputs, qualification=pin(qual['path']), qualification_logs=qlogs)
    manifest = read(study / 'manifest.json')
    require(set(manifest) == {'files'}, 'manifest schema')
    inventory, paths = manifest['files'], list(study.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'no symlinks')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(inventory) | {'manifest.json', 'receipt.json'}, 'complete output inventory')
    for name, value in inventory.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and descriptor(study / name) == value, 'payload pin: ' + name)
    receipt = read(study / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == PLAN_SHA
            and [receipt[k] for k in ('fits', 'fresh_fit_attempts', 'cached_fit_records', 'rows', 'raw_decodes',
                                     'reference_refits', 'legacy_refits', 'parent_refits', 'saved_fit_loads', 'saved_dev_loads', 'confirmation_access', 'official_test_access')]
            == [42, 6, 36, 184, 0, 0, 0, 0, 7, 2, False, False]
            and 0 < receipt['seconds'] <= cfg['wall_cap_seconds'] and receipt['clock'] == runtime['clock']
            and 0 < receipt['monotonic_seconds'] <= process['elapsed_seconds'], 'complete original closure')
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
    """Independent seven-family selection, 67 accuracy and two compute rules."""
    dev = cfg['partitions']['dev']
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    expected = {(name, arm, seed, rate, horizon) for name in dev for arm in ARMS
                for seed in SEEDS for rate in RATES for horizon in (64, 128)}
    expected |= {(name, arm, None, None, horizon) for name in dev for arm in REFS for horizon in (64, 128)}
    require(len(rows) == len(lookup) == 184 and set(lookup) == expected, 'complete unique metric roster')
    def metric(name, arm, seed=None, rate=None):
        row = lookup[name, arm, seed, rate, 128]
        value = row['metrics']
        if row['status'] != 'PASS' or value is None:
            return None
        numeric = [value[k] for k in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')]
        numeric += value['per_joint_rmse_deg']
        if not (len(value['per_joint_rmse_deg']) == 6 and type(value['scalars']) is int and value['scalars'] > 0
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in numeric)):
            return None
        return value
    def pooled(values):
        return math.sqrt(sum(v['standardized_sse'] for v in values) / sum(v['scalars'] for v in values)) if all(v is not None for v in values) else None
    options, chosen = {}, {}
    for arm in ARMS:
        options[arm], eligible = [], []
        for rate in RATES:
            score = pooled([metric(name, arm, seed, rate) for name in dev for seed in SEEDS])
            options[arm].append({'rate': rate, 'eligible': score is not None, 'pooled_rmse': score})
            if score is not None:
                eligible.append((score, rate))
        chosen[arm] = min(eligible)[1] if eligible else None
    ridge_options, eligible = [], []
    for arm in REFS[:2]:
        score = pooled([metric(name, arm) for name in dev])
        ridge_options.append({'arm': arm, 'eligible': score is not None, 'pooled_rmse': score})
        if score is not None:
            eligible.append((score, arm))
    ridge = min(eligible)[1] if eligible else None
    selection = {'selected_rates': chosen, 'options': options, 'ridge_options': ridge_options,
                 'selected_ridge': ridge, 'scope': 'same exposed DEV2; three seeds per rate; all36 parent fits reselected without refitting'}
    conditions, details = [], {}
    def add(name, flag):
        conditions.append({'name': name, 'passed': bool(flag)})
    add('all_selected_families_and_causal_ridge_eligible', all(rate is not None for rate in chosen.values()) and ridge is not None)
    for name in dev:
        neural = {arm: [metric(name, arm, seed, chosen[arm]) for seed in SEEDS] if chosen[arm] is not None else [] for arm in ARMS}
        means = {arm: sum(v['standardized_rmse'] for v in values)/3 for arm, values in neural.items() if len(values) == 3 and all(v is not None for v in values)}
        refs = {arm: metric(name, arm)['standardized_rmse'] for arm in REFS if metric(name, arm) is not None}
        details[name] = {'means': means, 'references': refs}
        candidate = means.get('householder12')
        add(name + '/mean_5pct/householder', candidate is not None and 'householder' in means and candidate <= .95*means['householder'])
        for i, seed in enumerate(SEEDS):
            left = neural['householder12'][i] if neural['householder12'] else None
            right = neural['householder'][i] if neural['householder'] else None
            add(f'{name}/seed{seed}_no_harm/householder', left is not None and right is not None and left['standardized_rmse'] <= right['standardized_rmse'])
        for arm in CACHED_ARMS[1:]:
            add(name + '/mean_within_2pct/' + arm, candidate is not None and arm in means and candidate <= 1.02*means[arm])
            for i, seed in enumerate(SEEDS):
                left = neural['householder12'][i] if neural['householder12'] else None
                right = neural[arm][i] if neural[arm] else None
                add(f'{name}/seed{seed}_within_5pct/{arm}', left is not None and right is not None and left['standardized_rmse'] <= 1.05*right['standardized_rmse'])
        for arm in REFS[2:]:
            add(name + '/mean_5pct/' + arm, candidate is not None and arm in refs and candidate <= .95*refs[arm])
        add(name + '/within_5pct_causal_ridge', candidate is not None and ridge in refs and candidate <= 1.05*refs[ridge])
        for j in range(6):
            available = 'householder12' in means and 'gru32' in means
            add(f'{name}/joint{j}_no_10pct_harm', available and sum(v['per_joint_rmse_deg'][j] for v in neural['householder12'])/3 <= 1.1*sum(v['per_joint_rmse_deg'][j] for v in neural['gru32'])/3)
    require(len(conditions) == 67, '67 accuracy conditions')
    accuracy = sum(c['passed'] for c in conditions)
    def latency(arm):
        subset = [r for r in resources if r['arm'] == arm]
        if len(subset) != 3 or {r['seed'] for r in subset} != set(SEEDS):
            return None
        values = [r['timing']['median_seconds'] for r in subset]
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            return None
        if not all(r['learning_rate'] == chosen[arm] for r in subset):
            return None
        return sorted(values)[1]
    left = latency('householder12')
    for arm in ('dense_bounded', 'dense_mlp'):
        right = latency(arm)
        add('at_most_105pct_' + arm + '_latency', left is not None and right is not None and left <= 1.05*right)
    passed = sum(c['passed'] for c in conditions)
    require(len(conditions) == 69, '69 conditions')
    return selection, {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == 69 else 'DO_NOT_ADVANCE_REFLECTION_CAPACITY',
                       'passed': passed, 'total': 69, 'accuracy': {'passed': accuracy, 'total': 67},
                       'compute': {'passed': passed-accuracy, 'total': 2}, 'conditions': conditions, 'details': details,
                       'scope': 'adaptive exposed-DEV capacity comparison; no new storage advantage, independent confirmation, official benchmark score or novel architecture claim'}


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
            'nonfinite structured transition output; no repair'), 'qualified finite failure type')


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_reflection_capacity_study as replay
    import torch
    torch.set_num_threads(1)
    cfg = plan['config']
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'fit_attempts': 42, 'fresh_fit_attempts': 6, 'cached_fit_records': 36, 'checkpoint_replays': 0,
              'prediction_files': 0, 'metric_rows': 184, 'saved_batches': 3, 'dev_target_windows': 0,
              'raw_mat_decodes': 0, 'confirmation_decodes': 0, 'official_test_decodes': 0, 'optimizer_updates': 0}
    expected = {'registration.json', 'runtime.json', 'normalizers.npz', 'linear.npz',
                'causal_ridge_1.npz', 'causal_ridge_100.npz', 'causal_ridge_1.json', 'causal_ridge_100.json',
                'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json', 'parent-normalizers.npz', 'parent-fits.json'} | {'sources/' + name for name in SOURCES}
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
    parent_norm = load('parent-normalizers.npz')
    ancestor_norm = load('normalizers.npz', parent=True)
    require(set(parent_norm) == set(ancestor_norm) == set(norm), 'all normalizer fields')
    for name in norm:
        equal(norm[name], parent_norm[name], 'same parent normalizer')
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
    parent_fits = read(study / 'parent-fits.json')
    require(descriptor(study / 'parent-fits.json') == {k: plan['parent_payloads']['fits.json'][k] for k in ('sha256', 'bytes')}, 'unchanged parent fit ledger')
    parent_expected = {f'{arm}-{seed}-lr{i}' for arm in CACHED_ARMS for seed in SEEDS for i in range(2)}
    require(len(parent_fits) == 36 and {row['key'] for row in parent_fits} == parent_expected, 'complete original parent ledger')
    parent_lookup = {row['key']: row for row in parent_fits}
    roster = [(arm, seed, rate, f'{arm}-{seed}-lr{i}') for seed in SEEDS for i, rate in enumerate(RATES) for arm in FRESH_ARMS]
    roster += [(row['arm'], row['seed'], row['learning_rate'], row['key']) for row in parent_fits]
    require(len(fits) == len(roster) == 42, 'all original attempts retained')
    models, initials, failure_count = {}, {}, 0
    for number, (fit, (arm, seed, rate, key)) in enumerate(zip(fits, roster, strict=True), 1):
        require((fit['arm'], fit['seed'], fit['learning_rate'], fit['key']) == (arm, seed, rate, key), 'attempt order')
        fit_keys = {'key', 'arm', 'seed', 'learning_rate', 'fit', 'resources', 'origin'}
        fresh = arm == 'householder12'
        if fresh:
            fit_keys |= {'effective_status', 'native_fit_seconds', 'native_fit_error', 'native_fit_cap_seconds', 'initial_pairing'}
            require(fit['origin'] == 'fresh' and fit['native_fit_cap_seconds'] == 1800.
                    and type(fit['native_fit_seconds']) in (int, float) and math.isfinite(fit['native_fit_seconds'])
                    and fit['native_fit_seconds'] > 0, 'native fresh-fit scope')
            late = fit['native_fit_seconds'] >= 1800.
            require(fit['effective_status'] == ('FAILED' if late else fit['fit']['status'])
                    and fit['native_fit_error'] == ({'type': 'FitFailure', 'message': 'native fit cap crossed including preservation'} if late else None), 'honest native terminal status')
        else:
            fit_keys |= {'parent_origin', 'parent_fit', 'parent_files'}
            original = parent_lookup[key]
            require(fit['origin'] == 'cached_parent' and fit['parent_fit'] == original
                    and fit['parent_origin'] == original['origin'] and fit['fit'] == original['fit']
                    and fit['resources'] == original['resources']
                    and fit['parent_files'] == {name: plan['parent_payloads'][key + '/' + name] for name in LEGACY_FILES}, 'unchanged cached fit lineage')
        require(set(fit) == fit_keys, 'fit record schema')
        expected.update({f'completed-fit-{number:02d}.json', key + '/fit-receipt.json', key + '/trace.json'})
        require(read(study / f'completed-fit-{number:02d}.json') == fit, 'immediate attempt receipt')
        receipt, trace = read(study / key / 'fit-receipt.json'), read(study / key / 'trace.json')
        require(receipt == fit['fit'], 'unchanged original fit receipt')
        validate_fit_receipt(receipt, trace, rate, 1800 if fresh else 600 if arm == 'legacy_instant' else 900)
        for name, value in receipt['files'].items():
            require(descriptor(study / key / name) == value, 'fit file pin')
        model = replay.model_for(arm, seed, linear)
        initial, final, optimizer = (load(key + '/' + suffix + '.npz') for suffix in ('initial', 'final', 'optimizer'))
        initials[key] = initial
        template = model.state_dict()
        require(set(initial) == set(final) == set(template), 'checkpoint state keys')
        for name, tensor in template.items():
            equal(initial[name], tensor.detach().numpy(), 'initial model factory join')
            require(final[name].shape == initial[name].shape and final[name].dtype == initial[name].dtype, 'final state schema')
        for name, _ in model.named_buffers():
            equal(final[name], initial[name], 'fixed model buffers')
        resource = fit['resources']
        count = sum(p.numel() for p in model.parameters())
        state = 50 if arm == 'gru32' else 12
        require(count == PARAMETERS[arm] and not list(model.named_buffers()), 'qualified parameter/buffer count')
        require(set(resource) == {'parameters', 'parameter_bytes', 'buffer_bytes', 'state_scalars', 'state_bytes',
                                  'normalizer_bytes', 'inactive_parameters', 'dtype', 'input_bytes', 'output_bytes', 'temporary_workspace'}
                and resource['inactive_parameters'] == (192 if arm == 'legacy_instant' else 0)
                and resource['dtype'] == 'float32' and resource['input_bytes'] == 9216
                and resource['output_bytes'] == 6144 and resource['temporary_workspace'] == 'not measured', 'all-fit resource schema')
        require([resource[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, 4*count, state, 4*state, 0, 192], 'all-attempt logical resources')
        parameters = dict(model.named_parameters())
        require(set(optimizer) <= {name + '/' + suffix for name in parameters for suffix in ('exp_avg', 'exp_avg_sq', 'step')}, 'Adam key ownership')
        for name, parameter in parameters.items():
            keys = {name + '/' + suffix for suffix in ('exp_avg', 'exp_avg_sq', 'step')}
            require(not (keys & set(optimizer)) or keys <= set(optimizer), 'whole Adam slots')
            for suffix in ('exp_avg', 'exp_avg_sq'):
                if name + '/' + suffix in optimizer:
                    require(optimizer[name + '/' + suffix].shape == tuple(parameter.shape) and optimizer[name + '/' + suffix].dtype == np.float32, 'Adam slot shape')
            if name + '/step' in optimizer:
                step = optimizer[name + '/step']
                require(step.shape == () and step.dtype == np.float32 and np.isfinite(step)
                        and float(step) in (len(trace), len(trace) + 1), 'Adam step agrees with completed/failed attempt')
        require(not optimizer or len(optimizer) == 3 * len(parameters), 'all Adam parameter slots or unstarted optimizer')
        require(not trace or optimizer, 'completed updates retain Adam')
        if receipt['status'] == 'PASS':
            require(len(optimizer) == 3 * len(parameters) and all(np.isfinite(v).all() for v in [*final.values(), *optimizer.values()]), 'finite successful state')
            for name in parameters:
                require(float(optimizer[name + '/step']) == 4096, 'Adam completed steps')
        effective = fit.get('effective_status', receipt['status'])
        if effective == 'PASS':
            model.load_state_dict({name: torch.from_numpy(value) for name, value in final.items()}, strict=True)
            model.eval()
            models[key] = model
        else:
            failure_count += 1
    for fit in fits[:6]:
        key = fit['key']
        previous_key = key.replace('householder12-', 'householder-', 1)
        current, previous = initials[key], initials[previous_key]
        require(list(current) == list(previous), 'paired initial parameter order')
        shared = 'cell.reflection_raw'
        require(current[shared].shape == (2, 12, 11) and previous[shared].shape == (2, 4, 11), 'paired reflector shapes')
        checks = {name: bool(np.array_equal(value[:, :4] if name == shared else value, previous[name])) for name,value in current.items()}
        require(all(checks.values()), 'exact common raw initialization')
        require(fit['initial_pairing'] == {'exact_common_parameters': checks, 'first_four_reflections_equal': checks[shared],
                'operator_claim': 'same .999I in real arithmetic only; no bitwise R4/R12 operator claim'}, 'initial pairing receipt')
    counts['initial_pairing_checks'] = 6
    barrier = read(study / 'checkpoint-barrier.json')
    require(barrier == {'fresh_fit_attempts': 6, 'cached_fit_records': 36, 'fit_records': 42, 'dev_loads_this_run': 0, 'dev_exposed_prior': True,
                       'checkpoints': {fit['key']: descriptor(study / fit['key'] / 'final.npz') for fit in fits}}, 'all-attempt checkpoint barrier')
    saved_result = read(study / 'results.json')
    require(saved_result['version'] == cfg['version'] and saved_result['config'] == cfg, 'results identity')
    rows = []
    for name, (batch, target) in dev.items():
        for fit in fits:
            key = fit['key']
            common = {k: fit[k] for k in ('arm', 'seed', 'learning_rate')}
            common.update(recording=name, fit_key=key)
            error, prediction = None, None
            filename = 'prediction-' + name + '-' + key + '.npz'
            if key not in models:
                error = {'type': 'FailedTrainingAttempt', 'effective_status': fit.get('effective_status', fit['fit']['status'])}
                require(not (study / filename).exists(), 'no prediction from failed fit')
            else:
                counts['checkpoint_replays'] += 1
                try:
                    with torch.no_grad():
                        replayed = replay.old.infer(models[key], batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) in ('nonfinite LPV output; no rollout clipping or repair', 'nonfinite structured transition output; no repair'), 'unexpected replay failure')
                    require(not (study / filename).exists(), 'failed guard has no array')
                    error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                else:
                    saved = load(filename)
                    require(set(saved) == {'prediction'}, 'prediction archive keys')
                    prediction = saved['prediction']
                    equal(prediction, replayed, 'exact same-runtime checkpoint replay')
                    counts['prediction_files'] += 1
            rows.extend(metric_rows(common, prediction, target, norm['q_std'], error, np))
        for arm in REFS:
            saved = load('prediction-' + name + '-' + arm + '.npz')
            require(set(saved) == {'prediction'}, 'reference prediction keys')
            prediction = saved['prediction']
            require(prediction.shape == target.shape and prediction.dtype == np.float64, 'reference prediction schema')
            counts['prediction_files'] += 1
            expected_prediction = reference_prediction(arm, coefficients, batch, np)
            equal(prediction, expected_prediction, 'independent reference prediction')
            counts['reference_replays'] = counts.get('reference_replays', 0) + 1
            error = None
            rows.extend(metric_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None}, prediction, target, norm['q_std'], error, np))
    close(rows, saved_result['rows'], 'independent metrics')
    resources = read(study / 'resources.json')
    selection, result = decisions(rows, resources, cfg)
    close(selection, saved_result['selection'], 'independent rate selection')
    close(result, saved_result['result'], 'independent 69 conditions')
    require(read(study / 'receipt.json')['scientific_result'] == result['status'], 'terminal scientific status')
    validate_resources(resources, fits, selection, rows, coefficients, np)
    require(set(read(study / 'manifest.json')['files']) == expected, 'exact dynamic evidence roster')
    counts.update(failed_fit_attempts=failure_count, passed_fit_attempts=42-failure_count,
                  failed_metric_rows=sum(r['status'] == 'FAILED' for r in rows), resource_rows=len(resources),
                  manifest_files=len(expected), condition_rows=69,
                  completed_training_updates=sum(fit['fit']['completed_updates'] for fit in fits if fit['origin'] == 'fresh'),
                  archived_training_updates=sum(fit['fit']['completed_updates'] for fit in fits if fit['origin'] == 'cached_parent'),
                  initial_checkpoints=42, final_checkpoints=42, optimizer_checkpoints=42)
    require(authenticate(study, run_receipt) == (plan, inputs), 'unchanged inputs after replay')
    return {'version': 'robot-reflection-capacity-audit-v1', 'status': 'PASS', 'agreement': True, 'inputs': inputs,
            'auditor': pin(__file__), 'study': str(study), 'registration_sha256': PLAN_SHA,
            'source_pins': plan['sources'], 'results': {'rows': rows, 'selection': selection, 'result': result},
            'resources': resources, 'counts': counts, 'seconds': time.monotonic()-started,
            'checks': {'complete_original_closure': True, 'saved_fit_normalization': True, 'paired_batches': True,
                       'independent_dev_targets': True, 'independent_metrics_selection_rules': True,
                       'exact_qualified_checkpoint_replay': True, 'all_failed_attempts_retained': True,
                       'legacy_and_references_byte_identical': True, 'all42_checkpoint_barrier': True},
            'scope': ['No raw MAT decode, CONFIRM/TEST access or optimizer updates.',
                      'Metrics, batch reconstruction, target alignment, eight reference forecasts and 69 rules are independently recomputed.',
                      'All six fresh fit records and 36 archived fit records are retained; archived updates are not charged as new training.',
                      'Checkpoint replay uses the frozen qualified model implementation, not an independent recurrence.',
                      'Saved FIT/DEV filtering provenance is authenticated, not independently re-filtered.',
                      'Training gradients/updates and timings are not replayed; traces and Adam ownership are checked.']}


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


def validate_resources(resources, fits, selection, rows, coefficients, np):
    selected = [fit for fit in fits if selection['selected_rates'][fit['arm']] == fit['learning_rate']]
    require(len(resources) == len(selected) + 4, 'selected resource roster')
    for row, fit in zip(resources[:len(selected)], selected, strict=True):
        require((row['arm'], row['seed'], row['learning_rate']) == (fit['arm'], fit['seed'], fit['learning_rate']), 'resource identity')
        close({k: row[k] for k in fit['resources']}, fit['resources'], 'fit resource join')
    for row in resources:
        arm = row['arm']
        neural = arm in ARMS
        count = PARAMETERS[arm] if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
        state = (50 if arm == 'gru32' else 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
        require([row[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, count*(4 if neural else 8), state, state*(4 if neural else 8), 16 if arm in REFS[:2] else 0, 192], 'numeric resource accounting')
        if neural:
            require(row['inactive_parameters'] == (192 if arm == 'legacy_instant' else 0), 'inactive parameter disclosure')
        require(row['input_bytes'] == 9216 and row['output_bytes'] == 6144 and row['temporary_workspace'] == 'not measured', 'resource scope')
        timing = row['timing']
        values = timing['seconds']
        require(len(values) == 20 and all(math.isfinite(v) and v > 0 for v in values), 'twenty timings')
        close(float(np.median(values)), timing['median_seconds'], 'timing median')
        close(float(np.percentile(values, 95)), timing['p95_seconds'], 'timing p95')
    require([r['arm'] for r in resources[len(selected):]] == list(REFS), 'reference resource order')


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


if __name__ == '__main__':
    main()
