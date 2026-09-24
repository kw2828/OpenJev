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
PLAN_SHA = '7bf116569bc141bf949b04b5bd79d465b2cd0acda098ffcf63cc26553725c4de'
COMMIT = 'bf499540b1c98f55213c67ec4d65cf56fe6a6411'
ARMS = ('chain_memory', 'rewired_memory', 'chain_instant', 'gru_residual', 'quadratic_ar2')
REFS = ('linear_frozen', 'quadratic_frozen', 'persistence', 'velocity', 'direct_ridge_1', 'direct_ridge_100')
SEEDS, RATES = (8101, 8102, 8103), (.0001, .001)
SOURCES = ('src/openjev/research/joint_coupling.py', 'src/openjev/research/industrial_robot_data.py',
           'scripts/robot_coupling_study.py', 'tests/test_joint_coupling.py',
           'tests/test_industrial_robot_data.py', 'tests/test_robot_coupling_study.py',
           'research/robot-coupling-protocol.md')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_coupling_study.py', '--registration',
           'research/robot-coupling-registration.json', '--output', 'output/robot-coupling-study-v1']


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
    """Metadata/opaque bytes only. No NumPy, Torch, MAT or checkpoint decode."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    require(study == ROOT / 'output/robot-coupling-study-v1', 'fixed original study')
    require(run_receipt == ROOT / 'output/robot-coupling-engineering-v1/run-process-01.json', 'original process')
    process = read(run_receipt)
    require(process['command'] == COMMAND and process['returncode'] == 0
            and process['prefit_commit'] == COMMIT and process['registration_sha256'] == PLAN_SHA
            and process['scope'] == 'One original measured-data campaign, no retries'
            and math.isfinite(process['elapsed_seconds']) and process['elapsed_seconds'] > 0, 'original process closed successfully')
    registration = ROOT / 'research/robot-coupling-registration.json'
    require(descriptor(registration)['sha256'] == PLAN_SHA, 'original registration pin')
    plan = read(registration)
    require(set(plan['sources']) == set(SOURCES), 'seven frozen sources')
    for name, value in plan['sources'].items():
        require(descriptor(ROOT / name) == value == descriptor(study / 'sources' / name), 'source/snapshot: ' + name)
    require(read(study / 'registration.json') == plan, 'saved registration semantic join')
    admission = read(study / 'admission.json')
    require(admission == {'registration_path': str(registration), 'registration_sha256': PLAN_SHA,
                         'sources': plan['sources'], 'confirmation_access': False, 'official_test_access': False}, 'admission')
    contract = plan['metadata_contract']
    require(descriptor(contract['path']) == {k: contract[k] for k in ('bytes', 'sha256')}, 'metadata contract pin')
    qual_path = Path(plan['qualification']['path'])
    require(descriptor(qual_path) == {k: plan['qualification'][k] for k in ('sha256', 'bytes')}, 'qualification pin')
    qual = read(qual_path)
    require(qual['status'] == 'PASS' and qual['sources'] == plan['sources'] and len(qual['runs']) == 2, 'qualified source closure')
    commands = [['.venv/bin/ruff', 'check', *SOURCES[:-1]],
                ['.venv/bin/python', '-m', 'pytest', '-q', *SOURCES[3:6]]]
    for entry, command in zip(qual['runs'], commands, strict=True):
        require(entry['command'] == command and entry['returncode'] == 0, 'qualification original command')
        # Original development receipt names the captured file and its digest.
        require(descriptor(ROOT / entry['log']) == entry['log_pin'], 'qualification log')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'single-thread audit environment')
    env = plan['environment']
    require(env['python'] == sys.version and env['machine'] == platform.machine()
            and env['platform'] == platform.platform(), 'same runtime/platform')
    for package in ('numpy', 'scipy', 'torch'):
        require(importlib.metadata.version(package) == env[package], 'same package: ' + package)
    manifest = read(study / 'manifest.json')
    require(set(manifest) == {'files'}, 'manifest schema')
    inventory = manifest['files']
    files = {str(p.relative_to(study)) for p in study.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in study.rglob('*')), 'no symlink evidence')
    require(files == set(inventory) | {'manifest.json', 'receipt.json'}, 'complete original inventory')
    for name, value in inventory.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(study / name) == value, 'payload pin: ' + name)
    receipt = read(study / 'receipt.json')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == PLAN_SHA
            and receipt['sources_before'] == receipt['sources_after'] == plan['sources']
            and [receipt[k] for k in ('fits', 'rows', 'fit_decodes', 'dev_decodes', 'confirmation_decodes', 'official_test_decodes')]
            == [30, 144, 7, 2, 0, 0], 'producer terminal closure/counts')
    require(0 < receipt['elapsed_seconds'] <= plan['config']['wall_cap_seconds'], 'original wall cap')
    inputs = {key: pin(path) for key, path in {
        'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'), 'registration': registration,
        'manifest': study / 'manifest.json', 'producer_receipt': study / 'receipt.json',
        'qualification': qual_path, 'metadata_contract': Path(contract['path'])}.items()}
    inputs['qualification_logs'] = [pin(ROOT / entry['log']) for entry in qual['runs']]
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


def decisions(rows, cfg):
    """Independent complete-roster selection and literal 55 registered conditions."""
    dev = cfg['partitions']['dev']
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    require(len(lookup) == 144 == len(rows), 'unique full metric roster')
    def metric(name, arm, seed=None, rate=None):
        return lookup[name, arm, seed, rate, 128]['metrics']
    def pooled(group):
        return math.sqrt(sum(v['standardized_sse'] for v in group) / sum(v['scalars'] for v in group)) if all(v is not None for v in group) else None
    options, chosen = {}, {}
    for arm in ARMS:
        candidates = []
        options[arm] = []
        for rate in RATES:
            value = pooled([metric(name, arm, seed, rate) for name in dev for seed in SEEDS])
            options[arm].append({'learning_rate': rate, 'eligible': value is not None, 'pooled_dev_h128_rmse': value})
            if value is not None:
                candidates.append((value, rate))
        chosen[arm] = min(candidates)[1] if candidates else None
    direct = []
    for arm in REFS[-2:]:
        value = pooled([metric(name, arm) for name in dev])
        direct.append({'arm': arm, 'eligible': value is not None, 'pooled_dev_h128_rmse': value})
    valid = [(r['pooled_dev_h128_rmse'], r['arm']) for r in direct if r['eligible']]
    selected_direct = min(valid)[1] if valid else None
    selection = {'selected_rates': chosen, 'rate_options': options, 'selected_direct': selected_direct,
                 'direct_options': direct, 'selection_scope': 'pooled internal DEV H128, no CONFIRM or official TEST access'}
    conditions, details = [], {}
    def add(name, flag):
        conditions.append({'name': name, 'passed': bool(flag)})
    add('all_mandatory_selected_recipes_finite', all(v is not None for v in chosen.values())
        and selected_direct is not None and all(metric(name, arm) is not None for name in dev for arm in REFS[:4]))
    for name in dev:
        neural = {arm: [metric(name, arm, seed, chosen[arm]) for seed in SEEDS] if chosen[arm] is not None else [] for arm in ARMS}
        means = {arm: sum(v['standardized_rmse'] for v in group) / 3 for arm, group in neural.items() if len(group) == 3 and all(v is not None for v in group)}
        refs = {arm: metric(name, arm)['standardized_rmse'] for arm in REFS if metric(name, arm) is not None}
        details[name] = {'selected_family_mean_standardized_rmse': means, 'reference_standardized_rmse': refs}
        candidate = means.get('chain_memory')
        controls = ('chain_instant', 'rewired_memory', 'gru_residual', 'quadratic_ar2')
        for arm in (*controls, *REFS[:4]):
            control = means.get(arm, refs.get(arm))
            add(name + '/mean_10pct/' + arm, candidate is not None and control is not None and candidate <= .9 * control)
        for arm in controls:
            for i, seed in enumerate(SEEDS):
                left = neural['chain_memory'][i] if neural['chain_memory'] else None
                right = neural[arm][i] if neural[arm] else None
                add(f'{name}/seed{seed}_5pct/{arm}', left is not None and right is not None and left['standardized_rmse'] <= .95 * right['standardized_rmse'])
        for j in range(6):
            available = 'chain_memory' in means and 'gru_residual' in means
            add(f'{name}/joint{j}_no_10pct_harm_vs_gru', available and
                sum(v['per_joint_rmse_deg'][j] for v in neural['chain_memory']) / 3 <= 1.1 * sum(v['per_joint_rmse_deg'][j] for v in neural['gru_residual']) / 3)
        direct_value = refs.get(selected_direct)
        add(name + '/within_5pct_direct_history', candidate is not None and direct_value is not None and candidate <= 1.05 * direct_value)
    passed = sum(v['passed'] for v in conditions)
    require(len(conditions) == 55, '55 conditions')
    return selection, {'status': 'ADVANCE_TO_SEPARATELY_AUTHORIZED_CONFIRMATION' if passed == 55 else 'DO_NOT_ADVANCE_COUPLING',
                       'passed': passed, 'total': 55, 'conditions': conditions, 'details': details,
                       'scope': 'internal DEV screening after two-rate selection, not held-out confirmation or official benchmark performance'}


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_coupling_study as replay
    import torch
    torch.set_num_threads(1)
    cfg = plan['config']
    counts = {'npz_decodes': 0, 'array_decodes': 0, 'fit_attempts': 30, 'checkpoint_replays': 0,
              'prediction_files': 0, 'metric_rows': 144, 'saved_batches': 3, 'dev_target_windows': 0,
              'raw_mat_decodes': 0, 'confirmation_decodes': 0, 'official_test_decodes': 0, 'optimizer_updates': 0}
    expected = {'registration.json', 'admission.json', 'normalizers.npz', 'references.npz', 'reference-fit.json',
                'checkpoint-barrier.json', 'results.json', 'resources.json', 'fits.json'} | {'sources/' + name for name in SOURCES}
    def load(name):
        expected.add(name)
        with np.load(study / name, allow_pickle=False) as data:
            value = {key: data[key].copy() for key in data.files}
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
            data = load(f'{partition}-data-{name}.npz')
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
        rng = np.random.Generator(np.random.PCG64(seed + 510000))
        ids = rng.integers(0, 7, size=(1024, 16), dtype=np.int64)
        starts = np.empty_like(ids)
        for index in np.ndindex(ids.shape):
            starts[index] = rng.integers(64, lengths[int(ids[index])] - 32 - 64 + 1)
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
    coefficients = load('references.npz')
    shapes = {'linear_frozen': (6, 25), 'quadratic_frozen': (6, 325), 'direct_ridge_1': (961, 768), 'direct_ridge_100': (961, 768)}
    require(set(coefficients) == set(shapes), 'reference roster')
    for key, shape in shapes.items():
        require(coefficients[key].shape == shape and coefficients[key].dtype == np.float64 and np.isfinite(coefficients[key]).all(), 'reference coefficient schema')
    fits = read(study / 'fits.json')
    roster = [(arm, seed, rate, f'{arm}-{seed}-lr{i}') for seed in SEEDS for i, rate in enumerate(RATES) for arm in ARMS]
    require(len(fits) == len(roster) == 30, 'all original attempts retained')
    models, failure_count = {}, 0
    for number, (fit, (arm, seed, rate, key)) in enumerate(zip(fits, roster, strict=True), 1):
        require((fit['arm'], fit['seed'], fit['learning_rate'], fit['key']) == (arm, seed, rate, key), 'attempt order')
        expected.update({f'completed-fit-{number:02d}.json', key + '/fit-receipt.json', key + '/trace.json'})
        require(read(study / f'completed-fit-{number:02d}.json') == fit, 'immediate attempt receipt')
        receipt = read(study / key / 'fit-receipt.json')
        require(receipt == fit['fit'] and receipt['status'] in ('PASS', 'FAILED'), 'fit receipt')
        require(fit['batches'] == descriptor(study / f'batches-{seed}.npz'), 'shared paired batch join')
        trace = read(study / key / 'trace.json')
        require(len(trace) == receipt['completed_updates'] <= 1024 and receipt['requested_updates'] == 1024 and receipt['learning_rate'] == rate, 'update ledger')
        for update, row in enumerate(trace, 1):
            require(set(row) == {'update', 'loss', 'gradient_norm_before_clip'} and row['update'] == update
                    and all(math.isfinite(row[k]) and row[k] >= 0 for k in ('loss', 'gradient_norm_before_clip')), 'finite completed trace')
        require(0 <= fit['construction_seconds'] <= receipt['fit_seconds'] and 0 <= receipt['optimizer_seconds'] <= receipt['fit_seconds'], 'timing scopes')
        require(set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json'}, 'fit evidence files')
        for name, value in receipt['files'].items():
            require(descriptor(study / key / name) == value, 'fit file pin')
        model = replay.model_for(arm, seed, coefficients['linear_frozen'], coefficients['quadratic_frozen'])
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
        state = 28 if arm in ('chain_memory', 'rewired_memory', 'gru_residual') else 18
        require([resource[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, 4*count, state, 4*state, 160 if arm in ARMS[:3] else 0, 192], 'all-attempt logical resources')
        parameters = dict(model.named_parameters())
        require(set(optimizer) <= {name + '/' + suffix for name in parameters for suffix in ('exp_avg', 'exp_avg_sq', 'step')}, 'Adam key ownership')
        for name, parameter in parameters.items():
            keys = {name + '/' + suffix for suffix in ('exp_avg', 'exp_avg_sq', 'step')}
            require(not (keys & set(optimizer)) or keys <= set(optimizer), 'whole Adam slots')
            for suffix in ('exp_avg', 'exp_avg_sq'):
                if name + '/' + suffix in optimizer:
                    require(optimizer[name + '/' + suffix].shape == tuple(parameter.shape) and optimizer[name + '/' + suffix].dtype == np.float32, 'Adam slot shape')
        if receipt['status'] == 'PASS':
            require(len(trace) == 1024 and receipt['error'] is None and receipt['fit_seconds'] <= 600, 'complete successful fit')
            require(len(optimizer) == 3 * len(parameters) and all(np.isfinite(v).all() for v in [*final.values(), *optimizer.values()]), 'finite successful state')
            for name in parameters:
                step = optimizer[name + '/step']
                require(step.shape == () and step.dtype == np.float32 and float(step) == 1024, 'Adam completed steps')
            model.load_state_dict({name: torch.from_numpy(value) for name, value in final.items()}, strict=True)
            model.eval()
            models[key] = model
        else:
            failure_count += 1
            require(isinstance(receipt['error'], dict) and set(receipt['error']) == {'type', 'message'}, 'retained failure reason')
            require(receipt['error']['type'] == 'FitFailure' or receipt['error'] == {
                'type': 'ValueError', 'message': 'nonfinite joint-coupling output; no clipping or repair'}, 'qualified finite failure type')
    barrier = read(study / 'checkpoint-barrier.json')
    require(barrier == {'fit_attempts': 30, 'dev_decodes': 0, 'confirmation_decodes': 0,
                       'final_checkpoints': {fit['key']: descriptor(study / fit['key'] / 'final.npz') for fit in fits},
                       'fit_statuses': {fit['key']: fit['fit']['status'] for fit in fits}}, 'all-attempt checkpoint barrier')
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
                error = {'type': 'FailedTrainingAttempt', 'message': 'not a completed trained model'}
                require(not (study / filename).exists(), 'no prediction from failed fit')
            else:
                counts['checkpoint_replays'] += 1
                try:
                    with torch.no_grad():
                        replayed = replay.infer(models[key], batch).numpy().astype(np.float64)
                except ValueError as exc:
                    require(str(exc) == 'nonfinite joint-coupling output; no clipping or repair', 'unexpected replay failure')
                    require(not (study / filename).exists(), 'failed guard has no array')
                    error = {'type': 'ValueError', 'message': str(exc)}
                else:
                    saved = load(filename)
                    require(set(saved) == {'prediction'}, 'prediction archive keys')
                    prediction = saved['prediction']
                    equal(prediction, replayed, 'exact same-runtime checkpoint replay')
                    counts['prediction_files'] += 1
                    if not np.isfinite(prediction).all():
                        error = {'type': 'ValueError', 'message': 'nonfinite DEV autoregression'}
            rows.extend(metric_rows(common, prediction, target, norm['q_std'], error, np))
        for arm in REFS:
            saved = load('prediction-' + name + '-' + arm + '.npz')
            require(set(saved) == {'prediction'}, 'reference prediction keys')
            prediction = saved['prediction']
            require(prediction.shape == target.shape and prediction.dtype == np.float64, 'reference prediction schema')
            counts['prediction_files'] += 1
            error = None if np.isfinite(prediction).all() else {'type': 'NonfiniteReference'}
            rows.extend(metric_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None}, prediction, target, norm['q_std'], error, np))
    close(rows, saved_result['rows'], 'independent metrics')
    selection, result = decisions(rows, cfg)
    close(selection, saved_result['selection'], 'independent rate selection')
    close(result, saved_result['result'], 'independent 55 conditions')
    require(read(study / 'receipt.json')['scientific_result'] == result['status'], 'terminal scientific status')
    resources = read(study / 'resources.json')
    validate_resources(resources, fits, selection, rows, coefficients, np)
    require(set(read(study / 'manifest.json')['files']) == expected, 'exact dynamic evidence roster')
    counts.update(failed_fit_attempts=failure_count, passed_fit_attempts=30-failure_count,
                  failed_metric_rows=sum(r['status'] == 'FAILED' for r in rows), resource_rows=len(resources),
                  manifest_files=len(expected), condition_rows=55,
                  completed_training_updates=sum(fit['fit']['completed_updates'] for fit in fits),
                  initial_checkpoints=30, final_checkpoints=30, optimizer_checkpoints=30)
    require(authenticate(study, run_receipt) == (plan, inputs), 'unchanged inputs after replay')
    return {'version': 'robot-coupling-audit-v1', 'status': 'PASS', 'agreement': True, 'inputs': inputs,
            'auditor': pin(__file__), 'study': str(study), 'registration_sha256': PLAN_SHA,
            'source_pins': plan['sources'], 'results': {'rows': rows, 'selection': selection, 'result': result},
            'resources': resources, 'counts': counts, 'seconds': time.monotonic()-started,
            'checks': {'complete_original_closure': True, 'saved_fit_normalization': True, 'paired_batches': True,
                       'independent_dev_targets': True, 'independent_metrics_selection_rules': True,
                       'exact_qualified_checkpoint_replay': True, 'all_failed_attempts_retained': True},
            'scope': ['No raw MAT decode, CONFIRM/TEST access or optimizer updates.',
                      'Metrics, batch reconstruction, target alignment and 55 rules are independently recomputed.',
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
                local = {'type': 'NonfiniteEvaluation', 'message': 'nonfinite metric arithmetic'}
        result.append({**common, 'horizon': horizon, 'status': 'PASS' if local is None else 'FAILED', 'error': local, 'metrics': value})
    return result


def validate_resources(resources, fits, selection, rows, coefficients, np):
    selected = [fit for fit in fits if selection['selected_rates'][fit['arm']] == fit['learning_rate']]
    refs = [*REFS[:4], *([selection['selected_direct']] if selection['selected_direct'] else [])]
    require(len(resources) == len(selected) + len(refs), 'selected resource roster')
    for row, fit in zip(resources[:len(selected)], selected, strict=True):
        require((row['arm'], row['seed'], row['learning_rate']) == (fit['arm'], fit['seed'], fit['learning_rate']), 'resource identity')
        close({k: row[k] for k in fit['resources']}, fit['resources'], 'resource fit join')
        require(row['optimizer_seconds'] == fit['fit']['optimizer_seconds'], 'loop time join')
    for row in resources:
        arm = row['arm']
        neural = arm in ARMS
        count = (266 if arm in ARMS[:3] else 1296 if arm == 'gru_residual' else 1950) if neural else int(coefficients[arm].size) if arm in coefficients else 0
        state = (28 if arm in ('chain_memory', 'rewired_memory', 'gru_residual') else 18) if neural else 18 if arm in REFS[:2] else 30 if arm == 'velocity' else 6 if arm == 'persistence' else 192
        require([row[k] for k in ('parameters', 'parameter_bytes', 'state_scalars', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
                == [count, count*(4 if neural else 8), state, state*(4 if neural else 8), 160 if arm in ARMS[:3] else 0, 192], 'independent logical resource accounting')
        require(row['request_input_bytes_float64'] == 9216 and row['request_output_bytes_float64'] == 6144
                and row['temporary_workspace'] == 'not measured', 'request/workspace scope')
        timing = row['timing']
        if timing is None:
            require(not neural and any(r['arm'] == arm and r['status'] == 'FAILED' for r in rows), 'missing timing only invalid reference')
        else:
            values = timing['seconds']
            require(len(values) == 20 and all(math.isfinite(v) and v > 0 for v in values), 'twenty timing samples')
            close(float(np.median(values)), timing['median_seconds'], 'timing median')
            close(float(np.percentile(values, 95)), timing['p95_seconds'], 'timing p95')
    require([r['arm'] for r in resources[len(selected):]] == refs, 'fixed resource roster')


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
