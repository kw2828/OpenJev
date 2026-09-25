"""Twelve-reflector capacity comparison using frozen parent data and controls.

Only six R12 fits are new. The original 36 fits and their artifacts are copied
unchanged, then every eligible family is retimed on the same complete request.
This is adaptive development on exposed DEV, not independent confirmation.
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
import robot_coupling_study as old
import robot_structured_study as parent
import torch
from torch import nn

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.reflection_capacity import ReflectionCapacityRobotTransition
from openjev.research.suspend_clock import SuspendClock

ROOT = old.ROOT
VERSION = 'robot-reflection-capacity-study-v1'
ARMS = ('householder12',)
ALL_ARMS = (*ARMS, *parent.ALL_ARMS)
REFERENCES = parent.REFERENCES
LEGACY_FILES = parent.LEGACY_FILES
PARENT_REGISTRATION = 'research/robot-structured-registration.json'
PARENT_FOLDER = ROOT / 'output/robot-structured-study-v1'
PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
            'causal_ridge_100.npz', 'causal_ridge_100.json', 'fits.json',
            *(f'batches-{seed}.npz' for seed in old.SEEDS),
            *(f'{arm}-{seed}-lr{ri}/{name}' for seed in old.SEEDS for ri in range(2)
              for arm in parent.ALL_ARMS for name in LEGACY_FILES))
SOURCES = (*parent.SOURCES, 'src/openjev/research/reflection_capacity.py',
           'tests/test_reflection_capacity.py', 'scripts/robot_reflection_capacity_study.py',
           'tests/test_robot_reflection_capacity_study.py', 'research/robot-reflection-capacity-protocol.md',
           'scripts/plot_robot_structured.py', 'scripts/audit_robot_structured.py',
           'src/openjev/research/suspend_clock.py')


def config():
    cfg = parent.config()
    cfg.update(version=VERSION, arms=list(ARMS), comparison_arms=list(ALL_ARMS),
               fit_cap_seconds=1800., wall_cap_seconds=10800., latency_ratio=1.05,
               r4_mean_ratio=.95, r4_paired_ratio=1., cached_parent_refit=False,
               reflections=12, parameters=806, state_scalars=12)
    cfg.pop('storage_ratio')
    return cfg


_pin = parent._pin
load_arrays = parent.load_arrays
resource_model = parent.resource_model
timed_request = parent.timed_request


def authenticate(registration):
    """Metadata and opaque hashes only; called before any saved-array decoder."""
    from plot_robot_structured import authenticate as authenticate_parent
    path = Path(registration).resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact capacity config required')
    old.require(set(plan['sources']) == set(SOURCES), 'exact capacity source roster required')
    old.require(all(os.environ.get(key) == '1' for key in old.THREADS), 'single-thread environment required')
    for name, pin in plan['sources'].items():
        old.require(old.descriptor(ROOT / name) == pin, 'capacity source changed: ' + name)
    prior = authenticate_parent(PARENT_FOLDER)
    prior_sha = old.descriptor(ROOT / PARENT_REGISTRATION)['sha256']
    old.require(prior_sha == plan['parent_registration_sha256'], 'parent registration changed')
    old.require(prior['plan']['data'] == plan['data'], 'unchanged inherited FIT/DEV data required')
    paths = {'manifest': PARENT_FOLDER / 'manifest.json', 'receipt': PARENT_FOLDER / 'receipt.json',
             'process': Path(prior['engineering']) / 'run-process-01.json',
             'audit': Path(prior['audit_path']),
             'audit_process': Path(prior['engineering']) / 'audit-process-01.json'}
    old.require(set(plan['parent_closure']) == set(paths), 'complete parent closure required')
    for key, path in paths.items():
        item = plan['parent_closure'][key]
        old.require(Path(item['path']) == path and _pin(item) == prior['inputs'][str(path)], 'parent closure join: ' + key)
    inventory = json.loads((PARENT_FOLDER / 'manifest.json').read_text())['files']
    old.require(set(plan['parent_payloads']) == set(PAYLOADS), 'complete190 cached parent payloads required')
    for name, item in plan['parent_payloads'].items():
        old.require(Path(item['path']) == PARENT_FOLDER / name, 'canonical parent payload path required')
        old.require(_pin(item) == inventory[name], 'parent payload manifest join: ' + name)
    _pin(plan['qualification'])
    qualification = json.loads(Path(plan['qualification']['path']).read_text())
    old.require(qualification['status'] == 'PASS' and qualification['sources'] == plan['sources'], 'qualified current sources required')
    old.require(bool(qualification['commands']), 'original qualification commands required')
    for row in qualification['commands']:
        old.require(row['returncode'] == 0 and old.descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification command/log')
    old.require(old.descriptor(ROOT / 'scripts/launch_robot_reflection_capacity.py') == plan['launcher'], 'launcher pin required')
    return plan, hashlib.sha256(raw).hexdigest()


def load_parent_arrays(plan, filename):
    old.require(filename in PAYLOADS and filename.endswith('.npz'), 'registered parent NPZ only')
    item = plan['parent_payloads'][filename]
    _pin(item)
    with np.load(item['path'], allow_pickle=False) as data:
        return {key: data[key].copy(order='K') for key in data.files}


def copy_parent(plan, filename, destination):
    old.require(filename in PAYLOADS, 'registered parent payload only')
    item = plan['parent_payloads'][filename]
    _pin(item)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(Path(item['path']).read_bytes())
    old.require(old.descriptor(destination) == {k: item[k] for k in ('sha256', 'bytes')}, 'exact copied parent bytes')


class ReflectionAdapter(parent.StructuredAdapter):
    def __init__(self, seed):
        nn.Module.__init__(self)
        self.cell = ReflectionCapacityRobotTransition(12, seed)


def model_for(arm, seed, linear):
    old.require(arm in ALL_ARMS, 'declared reflection capacity family required')
    return ReflectionAdapter(seed) if arm == 'householder12' else parent.model_for(arm, seed, linear)


def initial_pairing(model, original):
    """Exact common raw parameters only; no materialized operators or rollout."""
    current = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    old.require(list(current) == list(original), 'parent initial parameter names/order required')
    shared = 'cell.reflection_raw'
    old.require(current[shared].shape == (2, 12, 11) and original[shared].shape == (2, 4, 11), 'paired reflector shapes')
    checks = {name: bool(np.array_equal(value[:, :4] if name == shared else value, original[name]))
              for name, value in current.items()}
    old.require(all(checks.values()), 'parent R4 common initialization differs')
    return {'exact_common_parameters': checks, 'first_four_reflections_equal': checks[shared],
            'operator_claim': 'same .999I in real arithmetic only; no bitwise R4/R12 operator claim'}


def effective_status(fit_row):
    return fit_row.get('effective_status', fit_row['fit']['status'])


def check_deadline(clock, started_ns, seconds, *, whole):
    if clock.now_ns() - started_ns >= int(seconds * 1_000_000_000):
        error = old.WholeStudyTimeout if whole else old.FitFailure
        raise error('native whole-study deadline exceeded' if whole else 'native single-fit deadline exceeded')


def select(rows, cfg):
    options, selected = {}, {}
    for arm in ALL_ARMS:
        options[arm] = []
        for rate in cfg['learning_rates']:
            subset = [r for r in rows if r['arm'] == arm and r['learning_rate'] == rate and r['horizon'] == 128]
            expected = {(name, seed) for name in cfg['partitions']['dev'] for seed in cfg['seeds']}
            valid = len(subset) == len(expected) and {(r['recording'], r['seed']) for r in subset} == expected and all(old.valid_metric_row(r) for r in subset)
            score = float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset) / sum(r['metrics']['scalars'] for r in subset))) if valid else None
            options[arm].append({'rate': rate, 'eligible': valid, 'pooled_rmse': score})
        eligible = [o for o in options[arm] if o['eligible']]
        selected[arm] = min(eligible, key=lambda o: (o['pooled_rmse'], o['rate']))['rate'] if eligible else None
    ridge_options = []
    for arm in REFERENCES[:2]:
        subset = [r for r in rows if r['arm'] == arm and r['horizon'] == 128]
        eligible = len(subset) == 2 and {r['recording'] for r in subset} == set(cfg['partitions']['dev']) and all(old.valid_metric_row(r) for r in subset)
        score = float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset) / sum(r['metrics']['scalars'] for r in subset))) if eligible else None
        ridge_options.append({'arm': arm, 'eligible': eligible, 'pooled_rmse': score})
    eligible = [o for o in ridge_options if o['eligible']]
    return {'selected_rates': selected, 'options': options, 'ridge_options': ridge_options,
            'selected_ridge': min(eligible, key=lambda o: (o['pooled_rmse'], o['arm']))['arm'] if eligible else None,
            'scope': 'same exposed DEV2; three seeds per rate; all36 parent fits reselected without refitting'}


def evaluate_rule(rows, selection, resources, cfg):
    conditions, details = [], {}
    def add(name, passed):
        conditions.append({'name': name, 'passed': bool(passed)})
    # Recompute eligibility rather than trusting a caller-supplied selection.
    expected_selection = select(rows, cfg)
    old.require(selection == expected_selection, 'selection must match complete fixed row roster')
    rates = selection['selected_rates']
    reference_complete = all(len([r for r in rows if r['arm'] == arm and r['recording'] == name and r['horizon'] == 128]) == 1
                             for arm in REFERENCES for name in cfg['partitions']['dev'])
    add('all_selected_families_and_causal_ridge_eligible', reference_complete and
        all(rates[a] is not None for a in ALL_ARMS) and selection['selected_ridge'] is not None)
    for name in cfg['partitions']['dev']:
        current = [r for r in rows if r['recording'] == name and r['horizon'] == 128]
        neural = {}
        for arm in ALL_ARMS:
            subset = [r for r in current if r['arm'] == arm and r['learning_rate'] == rates[arm]]
            complete = len(subset) == len(cfg['seeds']) and {r['seed'] for r in subset} == set(cfg['seeds']) and all(old.valid_metric_row(r) for r in subset)
            neural[arm] = {r['seed']: r['metrics'] for r in subset} if complete else {}
        means = {a: float(np.mean([v['standardized_rmse'] for v in m.values()])) for a,m in neural.items() if m}
        refs = {}
        for arm in REFERENCES:
            subset = [r for r in current if r['arm'] == arm]
            if len(subset) == 1 and old.valid_metric_row(subset[0]):
                refs[arm] = subset[0]['metrics']['standardized_rmse']
        details[name] = {'means': means, 'references': refs}
        candidate = means.get('householder12')
        add(name + '/mean_5pct/householder', candidate is not None and 'householder' in means and candidate <= cfg['r4_mean_ratio'] * means['householder'])
        for seed in cfg['seeds']:
            left, right = neural['householder12'].get(seed), neural['householder'].get(seed)
            add(f'{name}/seed{seed}_no_harm/householder', left is not None and right is not None and left['standardized_rmse'] <= cfg['r4_paired_ratio'] * right['standardized_rmse'])
        for other in ALL_ARMS[2:]:
            add(name + '/mean_within_2pct/' + other, candidate is not None and other in means and candidate <= cfg['mean_ratio'] * means[other])
            for seed in cfg['seeds']:
                left, right = neural['householder12'].get(seed), neural[other].get(seed)
                add(f'{name}/seed{seed}_within_5pct/{other}', left is not None and right is not None and left['standardized_rmse'] <= cfg['paired_ratio'] * right['standardized_rmse'])
        for other in REFERENCES[2:]:
            add(name + '/mean_5pct/' + other, candidate is not None and other in refs and candidate <= .95 * refs[other])
        ridge = refs.get(selection['selected_ridge'])
        add(name + '/within_5pct_causal_ridge', candidate is not None and ridge is not None and candidate <= 1.05 * ridge)
        for joint in range(6):
            complete = set(neural['householder12']) == set(neural['gru32']) == set(cfg['seeds'])
            left = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['householder12'].values()]) if complete else None
            right = np.mean([m['per_joint_rmse_deg'][joint] for m in neural['gru32'].values()]) if complete else None
            add(f'{name}/joint{joint}_no_10pct_harm', complete and left <= 1.1 * right)
    accuracy_passed = sum(c['passed'] for c in conditions)
    old.require(len(conditions) == 67, 'exact67 accuracy conditions')
    def chosen_cost(arm):
        subset = [r for r in resources if r['arm'] == arm]
        if len(subset) != 3 or {r['seed'] for r in subset} != set(cfg['seeds']):
            return None
        values = [r['timing']['median_seconds'] for r in subset]
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            return None
        if not all(r['learning_rate'] == rates[arm] for r in subset):
            return None
        return float(np.median(values))
    left = chosen_cost('householder12')
    for other in ('dense_bounded', 'dense_mlp'):
        right = chosen_cost(other)
        add('at_most_105pct_' + other + '_latency', left is not None and right is not None and left <= cfg['latency_ratio'] * right)
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == 69 else 'DO_NOT_ADVANCE_REFLECTION_CAPACITY',
            'passed': passed, 'total': 69, 'accuracy': {'passed': accuracy_passed, 'total': 67},
            'compute': {'passed': passed - accuracy_passed, 'total': 2},
            'conditions': conditions, 'details': details,
            'scope': 'adaptive exposed-DEV capacity comparison; no new storage advantage, independent confirmation, official benchmark score or novel architecture claim'}


def run(registration, output):
    clock = SuspendClock()
    started_ns = clock.now_ns()
    started = time.monotonic()
    plan, sha = authenticate(registration)
    cfg = config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    models, fits, rows, resources = {}, [], [], []
    def check():
        check_deadline(clock, started_ns, cfg['wall_cap_seconds'], whole=True)
    try:
        check()
        old.write_json(output / 'runtime.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                       'platform': platform.platform(), 'machine': platform.machine(), 'host': platform.node(),
                       'thread_env': {key: os.environ.get(key) for key in old.THREADS}, 'torch_threads': torch.get_num_threads(),
                       'clock': clock.backend, 'timing_scope': 'parent complete physical request plus uniform native deadline check'})
        with (output / 'registration.json').open('xb') as handle:
            handle.write(Path(registration).read_bytes())
        for name in SOURCES:
            path = output / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            old.require(old.descriptor(path) == plan['sources'][name], 'source snapshot changed')
        physical_fit = [{'name': name, **load_arrays(plan, 'fit-data-' + name + '.npz')} for name in cfg['partitions']['fit']]
        norm = old.normalizers(physical_fit, cfg['skip'])
        prior_norm = load_arrays(plan, 'normalizers.npz')
        parent_norm = load_parent_arrays(plan, 'normalizers.npz')
        old.require(set(norm) == set(prior_norm) == set(parent_norm)
                    and all(np.array_equal(norm[k], prior_norm[k])
                            and np.array_equal(norm[k], parent_norm[k]) for k in norm), 'same FIT-only normalization')
        copy_parent(plan, 'normalizers.npz', output / 'normalizers.npz')
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        linear = load_parent_arrays(plan, 'linear.npz')['coefficient']
        copy_parent(plan, 'normalizers.npz', output / 'parent-normalizers.npz')
        copy_parent(plan, 'linear.npz', output / 'linear.npz')
        ridges = {}
        for penalty in cfg['direct_ridges']:
            arm = f'causal_ridge_{int(penalty)}'
            saved = load_parent_arrays(plan, arm + '.npz')
            item = plan['parent_payloads'][arm + '.json']
            _pin(item)
            metadata = json.loads(Path(item['path']).read_text())
            old.require(set(saved) == {f'h{h:03d}' for h in range(1,129)}, 'complete cached ridge bank')
            ridges[arm] = CausalRobotRidge(tuple(saved[f'h{h:03d}'] for h in range(1,129)), penalty, metadata['fit_rows'])
            old.require(ridges[arm].metadata() == {k: v for k,v in metadata.items() if k != 'fit_seconds'}, 'cached ridge metadata')
            copy_parent(plan, arm + '.npz', output / (arm + '.npz'))
            copy_parent(plan, arm + '.json', output / (arm + '.json'))
            check()
        for seed in cfg['seeds']:
            batches = load_parent_arrays(plan, f'batches-{seed}.npz')
            expected_batches = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(batches) == set(expected_batches) and all(np.array_equal(batches[k], expected_batches[k]) for k in batches), 'exact parent paired batches')
            copy_parent(plan, f'batches-{seed}.npz', output / f'batches-{seed}.npz')
            for ri, rate in enumerate(cfg['learning_rates']):
                for arm in ARMS:
                    check()
                    key = f'{arm}-{seed}-lr{ri}'
                    before = time.monotonic()
                    fit_started_ns = clock.now_ns()
                    def fit_check(native_start=fit_started_ns):
                        check()
                        check_deadline(clock, native_start, cfg['fit_cap_seconds'], whole=False)
                    model = model_for(arm, seed, linear)
                    original_initial = load_parent_arrays(plan, f'householder-{seed}-lr{ri}/initial.npz')
                    pairing = initial_pairing(model, original_initial)
                    fit = old.train_one(model, fit_data, batches, cfg=cfg, lr=rate, folder=output / key, check=fit_check, fit_started=before)
                    model.zero_grad(set_to_none=True)
                    native_seconds = (clock.now_ns() - fit_started_ns) / 1e9
                    # Preserve the original helper receipt even if preservation crossed the native cap.
                    effective = 'FAILED' if native_seconds >= cfg['fit_cap_seconds'] else fit['status']
                    models[key] = model
                    fits.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate, 'fit': fit,
                                 'effective_status': effective, 'native_fit_seconds': native_seconds,
                                 'native_fit_error': {'type': 'FitFailure', 'message': 'native fit cap crossed including preservation'} if native_seconds >= cfg['fit_cap_seconds'] else None,
                                 'native_fit_cap_seconds': cfg['fit_cap_seconds'], 'initial_pairing': pairing,
                                 'resources': resource_model(model, arm), 'origin': 'fresh'})
                    old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
                    check()
                    print(json.dumps({'completed': len(fits), 'key': key, 'status': effective, 'helper_status': fit['status'], 'updates': fit['completed_updates']}), flush=True)
        old.require(len(fits) == 6, 'all6 fresh attempts before this study DEV load')
        copy_parent(plan, 'fits.json', output / 'parent-fits.json')
        cached = json.loads((output / 'parent-fits.json').read_text())
        expected_keys = {f'{arm}-{seed}-lr{ri}' for arm in parent.ALL_ARMS for seed in cfg['seeds'] for ri in range(2)}
        old.require(len(cached) == 36 and {f['key'] for f in cached} == expected_keys, 'complete36 original cached records')
        for original in cached:
            key, arm, seed, rate = (original[k] for k in ('key', 'arm', 'seed', 'learning_rate'))
            old.require(arm in parent.ALL_ARMS and seed in cfg['seeds'] and rate in cfg['learning_rates']
                        and key == f"{arm}-{seed}-lr{cfg['learning_rates'].index(rate)}", 'cached fit identity')
            for name in LEGACY_FILES:
                copy_parent(plan, key + '/' + name, output / key / name)
            old.require(json.loads((output / key / 'fit-receipt.json').read_text()) == original['fit'], 'original cached receipt join')
            if original['fit']['status'] == 'PASS':
                model = model_for(arm, seed, linear)
                state = load_parent_arrays(plan, key + '/final.npz')
                model.load_state_dict({k: torch.from_numpy(v) for k,v in state.items()}, strict=True)
                model.zero_grad(set_to_none=True)
                old.require(resource_model(model, arm) == original['resources'], 'unchanged cached model resources')
                models[key] = model
            fits.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                         'fit': original['fit'], 'resources': original['resources'], 'origin': 'cached_parent',
                         'parent_origin': original['origin'], 'parent_fit': original,
                         'parent_files': {name: plan['parent_payloads'][key + '/' + name] for name in LEGACY_FILES}})
            old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
            check()
        old.require(len(fits) == 42, 'complete42 fresh and cached fit roster')
        old.write_json(output / 'checkpoint-barrier.json', {'fresh_fit_attempts': 6, 'cached_fit_records': 36, 'fit_records': 42,
                       'dev_loads_this_run': 0, 'dev_exposed_prior': True,
                       'checkpoints': {f['key']: old.descriptor(output / f['key'] / 'final.npz') for f in fits}})
        physical_timing = None
        for name in cfg['partitions']['dev']:
            physical = {'name': name, **load_arrays(plan, 'dev-data-' + name + '.npz')}
            data = old.normalized_record(physical, norm)
            starts = old.dev_windows(len(data['q']), cfg)
            choices = {'record': np.zeros(len(starts), dtype=np.int64), 'start': starts}
            batch = old.window_batch([data], choices, 32, 128)
            np.savez_compressed(output / ('dev-windows-' + name + '.npz'), starts=starts, target=batch['target'])
            if physical_timing is None:
                physical_timing = old.window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, 32, 128)
            for f in fits:
                prediction, error = None, None
                check()
                if effective_status(f) == 'PASS':
                    try:
                        with torch.no_grad():
                            prediction = old.infer(models[f['key']], batch).numpy().astype(np.float64)
                    except old.FitFailure as exc:
                        error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                    if prediction is not None:
                        np.savez_compressed(output / ('prediction-' + name + '-' + f['key'] + '.npz'), prediction=prediction)
                else:
                    error = {'type': 'FailedTrainingAttempt', 'effective_status': effective_status(f)}
                common = {'recording': name, 'arm': f['arm'], 'seed': f['seed'], 'learning_rate': f['learning_rate'], 'fit_key': f['key']}
                rows.extend(old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error))
            for arm in REFERENCES:
                prediction = predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges else old.reference_predict(arm, {'linear_frozen': linear}, batch)
                np.savez_compressed(output / ('prediction-' + name + '-' + arm + '.npz'), prediction=prediction)
                rows.extend(old.scored_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None}, prediction, batch['target'], norm['q_std'], cfg))
            check()
        old.require(len(rows) == 184, 'all184 candidate and reference metric rows')
        selection = select(rows, cfg)
        for f in fits:
            if selection['selected_rates'][f['arm']] == f['learning_rate']:
                old.require(all(p.grad is None for p in models[f['key']].parameters()), 'no retained gradients in inference storage')
                def pred(batch, key=f['key']):
                    check()
                    return old.infer(models[key], batch).numpy().astype(np.float64)
                resources.append({'arm': f['arm'], 'seed': f['seed'], 'learning_rate': f['learning_rate'], **f['resources'],
                                  'timing': timed_request(pred, physical_timing, norm, cfg)})
        for arm in REFERENCES:
            def pred(batch, name=arm):
                check()
                return predict_ridge(ridges[name], batch['q_context'], batch['u_context'], batch['future_u']) if name in ridges else old.reference_predict(name, {'linear_frozen': linear}, batch)
            count = ridges[arm].metadata()['coefficient_count'] if arm in ridges else (150 if arm == 'linear_frozen' else 0)
            state = 192 if arm in ridges else (18 if arm == 'linear_frozen' else 6)
            resources.append({'arm': arm, 'seed': None, 'parameters': count, 'parameter_bytes': count * 8,
                              'state_scalars': state, 'state_bytes': state * 8, 'normalizer_bytes': 192,
                              'buffer_bytes': 16 if arm in ridges else 0, 'dtype': 'float64',
                              'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured',
                              'timing': timed_request(pred, physical_timing, norm, cfg)})
        result = evaluate_rule(rows, selection, resources, cfg)
        old.write_json(output / 'results.json', {'version': VERSION, 'config': cfg, 'rows': rows, 'selection': selection, 'result': result})
        old.write_json(output / 'resources.json', resources)
        old.write_json(output / 'fits.json', fits)
        after, after_sha = authenticate(registration)
        old.require(after == plan and after_sha == sha, 'registration unchanged')
        old.write_json(output / 'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}})
        check()
        old.write_json(output / 'receipt.json', {'status': 'PASS', 'registration_sha256': sha, 'seconds': (clock.now_ns() - started_ns) / 1e9,
                       'clock': clock.backend, 'monotonic_seconds': time.monotonic() - started,
                       'fits': len(fits), 'fresh_fit_attempts': 6, 'cached_fit_records': 36, 'rows': len(rows), 'scientific_result': result['status'],
                       'raw_decodes': 0, 'reference_refits': 0, 'legacy_refits': 0, 'parent_refits': 0, 'saved_fit_loads': 7, 'saved_dev_loads': 2, 'confirmation_access': False, 'official_test_access': False})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        old.write_json(output / 'failure.json', {'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc),
                       'monotonic_seconds': time.monotonic() - started, 'completed_fits': len(fits), 'metric_rows': len(rows)})
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registration, args.output)
