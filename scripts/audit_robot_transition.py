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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = '8e12dac5ef2cc6acf95bea68bd8551f3040c103e628fa52697dba0511ed1c52a'
COMMIT = 'abf55fd00ce426763969eb6a240b7f3c26f7f043'
ARMS = ('lpv_recurrent', 'lpv_instant', 'lpv_constant', 'gru_residual')
REFS = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
SEEDS, RATES = (8101, 8102, 8103), (.001, .003)
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py',
           'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py',
           'tests/test_industrial_robot_data.py', 'tests/test_robot_coupling_study.py',
           'research/robot-coupling-protocol.md', 'src/openjev/research/bounded_robot_transition.py',
           'src/openjev/research/causal_robot_ridge.py', 'scripts/robot_transition_study.py',
           'tests/test_bounded_robot_transition.py', 'tests/test_causal_robot_ridge.py',
           'tests/test_robot_transition_study.py', 'research/robot-transition-protocol.md')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_transition_study.py', '--registration',
           'research/robot-transition-registration.json', '--output', 'output/robot-transition-study-v1']


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


def authenticate(study, run_receipt):
    """Only metadata/opaque hashing; all input admission precedes array decode."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    require(PLAN_SHA != 'UNBOUND' and COMMIT != 'UNBOUND', 'bind original registration and prefit commit first')
    require(study == ROOT / 'output/robot-transition-study-v1', 'fixed original study')
    require(run_receipt == ROOT / 'output/robot-transition-engineering-v1/run-process-01.json', 'original process path')
    process = read(run_receipt)
    require(process['command'] == COMMAND and process['returncode'] == 0
            and process['prefit_commit'] == COMMIT and process['registration_sha256'] == PLAN_SHA
            and math.isfinite(process['elapsed_seconds']) and process['elapsed_seconds'] > 0, 'successful original process')
    registration = ROOT / 'research/robot-transition-registration.json'
    require(descriptor(registration)['sha256'] == PLAN_SHA, 'frozen registration')
    plan = read(registration)
    require(plan['version'] == 'robot-transition-study-v1' and set(plan['sources']) == set(SOURCES), 'source roster')
    for name, value in plan['sources'].items():
        require(descriptor(ROOT / name) == value == descriptor(study / 'sources' / name), 'source/snapshot: ' + name)
    require(read(study / 'registration.json') == plan, 'saved registration')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread environment')
    env = plan['environment']
    require(env['python'] == sys.version and env['platform'] == platform.platform(), 'same runtime/platform')
    for package in ('numpy', 'scipy', 'torch'):
        require(importlib.metadata.version(package) == env[package], 'same package: ' + package)
    inputs = {key: pin(path) for key, path in {
        'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'), 'registration': registration,
        'manifest': study / 'manifest.json', 'producer_receipt': study / 'receipt.json'}.items()}
    closure = plan['parent_closure']
    require(set(closure) == {'manifest', 'receipt', 'process', 'audit', 'audit_process'}, 'closed parent roster')
    parent = {}
    for key, item in closure.items():
        require(descriptor(item['path']) == {k: item[k] for k in ('bytes', 'sha256')}, 'parent pin: ' + key)
        parent[key] = read(item['path'])
    require(parent['receipt']['status'] == 'PASS' and parent['process']['returncode'] == 0
            and parent['audit_process']['returncode'] == 0 and parent['audit']['status'] == 'PASS'
            and parent['audit']['agreement'] is True
            and parent['receipt']['registration_sha256'] == plan['parent_registration_sha256'], 'closed parent')
    expected_data = {f'{split}-data-{name}.npz' for split in ('fit', 'dev') for name in plan['config']['partitions'][split]}
    expected_data |= {'normalizers.npz', 'references.npz'}
    require(set(plan['data']) == expected_data, 'saved parent data roster')
    for name, item in plan['data'].items():
        expected = {k: item[k] for k in ('bytes', 'sha256')}
        require(descriptor(item['path']) == expected == parent['manifest']['files'][name], 'parent input join: ' + name)
    qual = plan['qualification']
    require(descriptor(qual['path']) == {k: qual[k] for k in ('bytes', 'sha256')}, 'qualification pin')
    qualification = read(qual['path'])
    commands = [['.venv/bin/ruff', 'check', SOURCES[9], SOURCES[7], SOURCES[8], *SOURCES[10:13]],
                ['.venv/bin/python', '-m', 'pytest', '-q', *SOURCES[3:6], *SOURCES[10:13]]]
    require(qualification['status'] == 'PASS' and len(qualification['commands']) == 2, 'qualified sources')
    for row, command in zip(qualification['commands'], commands, strict=True):
        require(row['command'] == command and row['returncode'] == 0
                and descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification logs/argv')
    inputs.update(parent_closure=closure, parent_data=plan['data'], qualification=pin(qual['path']))
    manifest = read(study / 'manifest.json')
    require(set(manifest) == {'files'}, 'manifest schema')
    inventory = manifest['files']
    paths = list(study.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'no symlinks')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(inventory) | {'manifest.json', 'receipt.json'}, 'complete output inventory')
    for name, value in inventory.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(study / name) == value, 'payload pin: ' + name)
    receipt = read(study / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == PLAN_SHA
            and [receipt[k] for k in ('fits', 'rows', 'raw_decodes', 'saved_fit_loads', 'saved_dev_loads', 'confirmation_access', 'official_test_access')]
            == [24, 112, 0, 7, 2, False, False]
            and 0 < receipt['seconds'] <= plan['config']['wall_cap_seconds'], 'complete original closure')
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
    """Independent complete-roster selection and all45 Boolean conditions."""
    dev = cfg['partitions']['dev']
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    require(len(lookup) == 112 == len(rows), 'complete unique metric roster')
    def metric(name, arm, seed=None, rate=None):
        return lookup[name, arm, seed, rate, 128]['metrics']
    def pooled(values):
        return math.sqrt(sum(v['standardized_sse'] for v in values) / sum(v['scalars'] for v in values)) if all(v is not None for v in values) else None
    options, chosen = {}, {}
    for arm in ARMS:
        options[arm], valid = [], []
        for rate in RATES:
            score = pooled([metric(name, arm, seed, rate) for name in dev for seed in SEEDS])
            options[arm].append({'rate': rate, 'eligible': score is not None, 'pooled_rmse': score})
            if score is not None:
                valid.append((score, rate))
        chosen[arm] = min(valid)[1] if valid else None
    ridge_options, valid = [], []
    for arm in REFS[:2]:
        score = pooled([metric(name, arm) for name in dev])
        ridge_options.append({'arm': arm, 'eligible': score is not None, 'pooled_rmse': score})
        if score is not None:
            valid.append((score, arm))
    ridge = min(valid)[1] if valid else None
    selection = {'selected_rates': chosen, 'options': options, 'ridge_options': ridge_options,
                 'selected_ridge': ridge, 'scope': 'same already-exposed DEV2; three individual fits, no seed selection or ensemble'}
    conditions, details = [], {}
    def add(name, flag):
        conditions.append({'name': name, 'passed': bool(flag)})
    add('all_selected_families_and_causal_ridge_eligible', all(rate is not None for rate in chosen.values()) and ridge is not None)
    for name in dev:
        neural = {arm: [metric(name, arm, seed, chosen[arm]) for seed in SEEDS] if chosen[arm] is not None else [] for arm in ARMS}
        means = {arm: sum(v['standardized_rmse'] for v in values)/3 for arm, values in neural.items() if len(values) == 3 and all(v is not None for v in values)}
        refs = {arm: metric(name, arm)['standardized_rmse'] for arm in REFS if metric(name, arm) is not None}
        details[name] = {'means': means, 'references': refs}
        candidate = means.get('lpv_recurrent')
        for arm in ARMS[1:]:
            add(name + '/mean_5pct/' + arm, candidate is not None and arm in means and candidate <= .95*means[arm])
            for i, seed in enumerate(SEEDS):
                left = neural['lpv_recurrent'][i] if neural['lpv_recurrent'] else None
                right = neural[arm][i] if neural[arm] else None
                add(f'{name}/seed{seed}_2pct/{arm}', left is not None and right is not None and left['standardized_rmse'] <= .98*right['standardized_rmse'])
        for arm in REFS[2:]:
            add(name + '/mean_5pct/' + arm, candidate is not None and arm in refs and candidate <= .95*refs[arm])
        add(name + '/within_5pct_causal_ridge', candidate is not None and ridge in refs and candidate <= 1.05*refs[ridge])
        for j in range(6):
            available = 'lpv_recurrent' in means and 'gru_residual' in means
            add(f'{name}/joint{j}_no_10pct_harm', available and sum(v['per_joint_rmse_deg'][j] for v in neural['lpv_recurrent'])/3 <= 1.1*sum(v['per_joint_rmse_deg'][j] for v in neural['gru_residual'])/3)
    def cost(arm):
        subset = [r for r in resources if r['arm'] == arm]
        if len(subset) != 3 or {r['seed'] for r in subset} != set(SEEDS):
            return None
        medians = sorted(r['timing']['median_seconds'] for r in subset)
        return medians[1], max(sum(r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) for r in subset)
    left, right = cost('lpv_recurrent'), cost('gru_residual')
    add('at_most_twice_gru_latency', left is not None and right is not None and left[0] <= 2*right[0])
    add('at_most_gru_numeric_storage', left is not None and right is not None and left[1] <= right[1])
    passed = sum(c['passed'] for c in conditions)
    require(len(conditions) == 45, '45 conditions')
    return selection, {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == 45 else 'DO_NOT_ADVANCE_TRANSITION',
                       'passed': passed, 'total': 45, 'conditions': conditions, 'details': details,
                       'scope': 'adaptive development result, not independent confirmation, official benchmark score or novel architecture evidence'}


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_transition_study as replay
    import torch
    torch.set_num_threads(1)
    cfg = plan['config']
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'fit_attempts': 24, 'checkpoint_replays': 0,
              'prediction_files': 0, 'metric_rows': 112, 'saved_batches': 3, 'dev_target_windows': 0,
              'raw_mat_decodes': 0, 'confirmation_decodes': 0, 'official_test_decodes': 0, 'optimizer_updates': 0}
    expected = {'registration.json', 'normalizers.npz', 'linear.npz',
                'causal_ridge_1.npz', 'causal_ridge_100.npz', 'causal_ridge_1.json', 'causal_ridge_100.json',
                'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json'} | {'sources/' + name for name in SOURCES}
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
    roster = [(arm, seed, rate, f'{arm}-{seed}-lr{i}') for seed in SEEDS for i, rate in enumerate(RATES) for arm in ARMS]
    require(len(fits) == len(roster) == 24, 'all original attempts retained')
    models, failure_count = {}, 0
    for number, (fit, (arm, seed, rate, key)) in enumerate(zip(fits, roster, strict=True), 1):
        require((fit['arm'], fit['seed'], fit['learning_rate'], fit['key']) == (arm, seed, rate, key), 'attempt order')
        expected.update({f'completed-fit-{number:02d}.json', key + '/fit-receipt.json', key + '/trace.json'})
        require(read(study / f'completed-fit-{number:02d}.json') == fit, 'immediate attempt receipt')
        receipt = read(study / key / 'fit-receipt.json')
        require(receipt == fit['fit'] and receipt['status'] in ('PASS', 'FAILED'), 'fit receipt')
        trace = read(study / key / 'trace.json')
        require(len(trace) == receipt['completed_updates'] <= 4096 and receipt['requested_updates'] == 4096 and receipt['learning_rate'] == rate, 'update ledger')
        for update, row in enumerate(trace, 1):
            require(set(row) == {'update', 'loss', 'gradient_norm_before_clip'} and row['update'] == update
                    and all(math.isfinite(row[k]) and row[k] >= 0 for k in ('loss', 'gradient_norm_before_clip')), 'finite completed trace')
        require(0 <= receipt['optimizer_seconds'] <= receipt['fit_seconds'], 'timing scopes')
        require(set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json'}, 'fit evidence files')
        for name, value in receipt['files'].items():
            require(descriptor(study / key / name) == value, 'fit file pin')
        model = replay.model_for(arm, seed, linear)
        initial, final, optimizer = (load(key + '/' + suffix + '.npz') for suffix in ('initial', 'final', 'optimizer'))
        template = model.state_dict()
        require(set(initial) == set(final) == set(template), 'checkpoint state keys')
        for name, tensor in template.items():
            equal(initial[name], tensor.detach().numpy(), 'initial model factory join')
            require(final[name].shape == initial[name].shape and final[name].dtype == initial[name].dtype, 'final state schema')
        for name, _ in model.named_buffers():
            equal(final[name], initial[name], 'fixed model buffers')
        resource = fit['resources']
        count = sum(p.numel() for p in model.parameters())
        state = 28 if arm == 'gru_residual' else 20 if arm == 'lpv_recurrent' else 12
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
        if receipt['status'] == 'PASS':
            require(len(trace) == 4096 and receipt['error'] is None and receipt['fit_seconds'] <= 600, 'complete successful fit')
            require(len(optimizer) == 3 * len(parameters) and all(np.isfinite(v).all() for v in [*final.values(), *optimizer.values()]), 'finite successful state')
            for name in parameters:
                step = optimizer[name + '/step']
                require(step.shape == () and step.dtype == np.float32 and float(step) == 4096, 'Adam completed steps')
            model.load_state_dict({name: torch.from_numpy(value) for name, value in final.items()}, strict=True)
            model.eval()
            models[key] = model
        else:
            failure_count += 1
            require(isinstance(receipt['error'], dict) and set(receipt['error']) == {'type', 'message'}, 'retained failure reason')
            require(receipt['error']['type'] == 'FitFailure' or receipt['error'] == {
                'type': 'ValueError', 'message': 'nonfinite LPV output; no rollout clipping or repair'}, 'qualified finite failure type')
    barrier = read(study / 'checkpoint-barrier.json')
    require(barrier == {'fit_attempts': 24, 'dev_loads_this_run': 0, 'dev_exposed_prior': True,
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
                error = {'type': 'FailedTrainingAttempt'}
                require(not (study / filename).exists(), 'no prediction from failed fit')
            else:
                counts['checkpoint_replays'] += 1
                try:
                    with torch.no_grad():
                        replayed = replay.old.infer(models[key], batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) == 'nonfinite LPV output; no rollout clipping or repair', 'unexpected replay failure')
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
    close(result, saved_result['result'], 'independent 45 conditions')
    require(read(study / 'receipt.json')['scientific_result'] == result['status'], 'terminal scientific status')
    validate_resources(resources, fits, selection, rows, coefficients, np)
    require(set(read(study / 'manifest.json')['files']) == expected, 'exact dynamic evidence roster')
    counts.update(failed_fit_attempts=failure_count, passed_fit_attempts=24-failure_count,
                  failed_metric_rows=sum(r['status'] == 'FAILED' for r in rows), resource_rows=len(resources),
                  manifest_files=len(expected), condition_rows=45,
                  completed_training_updates=sum(fit['fit']['completed_updates'] for fit in fits),
                  initial_checkpoints=24, final_checkpoints=24, optimizer_checkpoints=24)
    require(authenticate(study, run_receipt) == (plan, inputs), 'unchanged inputs after replay')
    return {'version': 'robot-transition-audit-v1', 'status': 'PASS', 'agreement': True, 'inputs': inputs,
            'auditor': pin(__file__), 'study': str(study), 'registration_sha256': PLAN_SHA,
            'source_pins': plan['sources'], 'results': {'rows': rows, 'selection': selection, 'result': result},
            'resources': resources, 'counts': counts, 'seconds': time.monotonic()-started,
            'checks': {'complete_original_closure': True, 'saved_fit_normalization': True, 'paired_batches': True,
                       'independent_dev_targets': True, 'independent_metrics_selection_rules': True,
                       'exact_qualified_checkpoint_replay': True, 'all_failed_attempts_retained': True},
            'scope': ['No raw MAT decode, CONFIRM/TEST access or optimizer updates.',
                      'Metrics, batch reconstruction, target alignment, causal ridge predictions and45 rules are independently recomputed.',
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
        count = (1296 if arm == 'gru_residual' else 468 if arm == 'lpv_constant' else 1014) if neural else 445440 if arm in REFS[:2] else 150 if arm == 'linear_frozen' else 0
        state = (28 if arm == 'gru_residual' else 20 if arm == 'lpv_recurrent' else 12) if neural else 192 if arm in REFS[:2] else 18 if arm == 'linear_frozen' else 6
        require([row[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, count*(4 if neural else 8), state, state*(4 if neural else 8), 16 if arm in REFS[:2] else 0, 192], 'numeric resource accounting')
        if neural:
            require(row['inactive_parameters'] == (192 if arm == 'lpv_instant' else 0), 'inactive parameter disclosure')
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
    require(not args.output.exists(), 'fresh audit output only')
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
