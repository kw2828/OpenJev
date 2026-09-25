"""Prospective observed-prefix initialization comparison on exposed DEV.

Eighteen fresh attempts precede all current DEV reads. Thirty previously fitted
controls retain their original evidence and are replayed without refitting.
No raw MAT, CONFIRM or official TEST data are decoded by this runner.
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
import robot_structured_study as structured
import robot_transition_study as transition
import torch

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.robot_history_initializer import HistoryInitializedDense
from openjev.research.suspend_clock import SuspendClock

ROOT = old.ROOT
VERSION = 'robot-history-initialization-study-v1'
ARMS = ('last_two', 'local_affine', 'temporal_affine')
CACHED_ARMS = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ALL_ARMS = (*ARMS, *CACHED_ARMS)
REFERENCES = transition.REFERENCES
LEGACY_FILES = structured.LEGACY_FILES
PERMUTATION = (*range(29, -1, -1), 30, 31)
PARENT_FOLDERS = {name: ROOT / f'output/robot-{name}-study-v1' for name in ('structured', 'transition')}
PARENT_REGISTRATIONS = {name: f'research/robot-{name}-registration.json' for name in PARENT_FOLDERS}
COMMON_PAYLOADS = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                   'causal_ridge_100.npz', 'causal_ridge_100.json',
                   *(f'batches-{seed}.npz' for seed in old.SEEDS))
PAYLOADS = {
    'structured': (*COMMON_PAYLOADS, 'fits.json', 'results.json',
                   *(f'{arm}-{seed}-lr{ri}/{name}' for arm in CACHED_ARMS[:-1]
                     for seed in old.SEEDS for ri in range(2) for name in LEGACY_FILES)),
    'transition': ('fits.json', 'results.json',
                   *(f'gru_residual-{seed}-lr{ri}/{name}' for seed in old.SEEDS for ri in range(2)
                     for name in LEGACY_FILES)),
}
SOURCES = (*structured.SOURCES, 'src/openjev/research/robot_history_initializer.py',
           'tests/test_robot_history_initializer.py', 'scripts/robot_history_initialization_study.py',
           'tests/test_robot_history_initialization_study.py', 'research/robot-history-initialization-protocol.md',
           'scripts/plot_robot_structured.py', 'scripts/audit_robot_structured.py',
           'src/openjev/research/suspend_clock.py')


def config():
    cfg = structured.config()
    for key in ('mean_ratio', 'paired_ratio', 'storage_ratio', 'joint_harm_ratio', 'direct_noninferiority_ratio'):
        cfg.pop(key, None)
    cfg.update(version=VERSION, arms=list(ARMS), comparison_arms=list(ALL_ARMS),
               fit_cap_seconds=1800., wall_cap_seconds=14400., latency_ratio=1.25,
               mean_reduction=.05, file_harm_ratio=1.02, cached_parent_refit=False,
               older_permutation=list(PERMUTATION), diagnostic_gate=False)
    return cfg


_pin = structured._pin
load_arrays = structured.load_arrays
timed_request = structured.timed_request


def authenticate(registration):
    """All original closures, sources and opaque inputs before any NPZ decoder."""
    from plot_robot_structured import authenticate as authenticate_structured
    raw = Path(registration).read_bytes()
    plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact history config required')
    old.require(set(plan['sources']) == set(SOURCES), 'exact history source roster required')
    old.require(all(os.environ.get(k) == '1' for k in old.THREADS), 'single-thread environment required')
    for name, pin in plan['sources'].items():
        old.require(old.descriptor(ROOT / name) == pin, 'history source changed: ' + name)
    prior = authenticate_structured(PARENT_FOLDERS['structured'])
    # This qualified metadata helper additionally binds the transition audit's
    # original process/output/manifest/data joins, including its full roster.
    prior_plan, prior_sha = structured.authenticate(ROOT / PARENT_REGISTRATIONS['structured'])
    old.require(prior_plan == prior['plan'] and plan['data'] == prior_plan['data'], 'same inherited data required')
    old.require(set(plan['parent_registration_sha256']) == set(PARENT_FOLDERS)
                and set(plan['parent_closure']) == set(PARENT_FOLDERS)
                and set(plan['parent_payloads']) == set(PARENT_FOLDERS), 'both parent groups required')
    paths = {'manifest': PARENT_FOLDERS['structured'] / 'manifest.json',
             'receipt': PARENT_FOLDERS['structured'] / 'receipt.json',
             'process': Path(prior['engineering']) / 'run-process-01.json',
             'audit': Path(prior['audit_path']),
             'audit_process': Path(prior['engineering']) / 'audit-process-01.json'}
    old.require(plan['parent_registration_sha256']['structured'] == prior_sha, 'structured registration join')
    old.require(plan['parent_registration_sha256']['transition'] == prior_plan['parent_registration_sha256'],
                'transition registration join')
    for group, folder in PARENT_FOLDERS.items():
        closure = plan['parent_closure'][group]
        old.require(set(closure) == set(paths), 'complete parent closure: ' + group)
        if group == 'structured':
            for key, path in paths.items():
                old.require(Path(closure[key]['path']) == path
                            and _pin(closure[key]) == prior['inputs'][str(path)], 'structured closure join: ' + key)
        else:
            old.require(closure == prior_plan['parent_closure'], 'exact original transition closure required')
            for item in closure.values():
                _pin(item)
        inventory = json.loads(Path(closure['manifest']['path']).read_text())['files']
        payloads = plan['parent_payloads'][group]
        old.require(set(payloads) == set(PAYLOADS[group]), 'complete parent payload roster: ' + group)
        for name, item in payloads.items():
            old.require(Path(item['path']) == folder / name
                        and _pin(item) == inventory[name], 'original parent payload join: ' + group + '/' + name)
    _pin(plan['qualification'])
    qualification = json.loads(Path(plan['qualification']['path']).read_text())
    old.require(qualification['status'] == 'PASS' and qualification['sources'] == plan['sources']
                and qualification['launcher'] == plan['launcher'] and qualification['commands'], 'qualified sources/launcher required')
    for row in qualification['commands']:
        old.require(row['returncode'] == 0 and old.descriptor(row['log'])['sha256'] == row['sha256'], 'qualification command/log join')
    old.require(old.descriptor(ROOT / 'scripts/launch_robot_history_initialization.py') == plan['launcher'], 'launcher changed')
    return plan, hashlib.sha256(raw).hexdigest()


def load_parent_arrays(plan, group, filename):
    old.require(group in PAYLOADS and filename in PAYLOADS[group] and filename.endswith('.npz'), 'registered parent NPZ required')
    item = plan['parent_payloads'][group][filename]
    _pin(item)
    with np.load(item['path'], allow_pickle=False) as data:
        return {key: data[key].copy(order='K') for key in data.files}


def copy_parent(plan, group, filename, destination):
    old.require(group in PAYLOADS and filename in PAYLOADS[group], 'registered parent payload required')
    item = plan['parent_payloads'][group][filename]
    _pin(item)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(Path(item['path']).read_bytes())
    old.require(old.descriptor(destination) == {k: item[k] for k in ('sha256', 'bytes')}, 'unchanged parent copy required')


class HistoryAdapter(HistoryInitializedDense):
    """Translate only generated numerical failures to preserved failed fits."""
    @staticmethod
    def _numeric_call(function, *args):
        try:
            return function(*args)
        except ValueError as exc:
            if str(exc) not in ('nonfinite structured transition output; no repair',
                                'finite CPU tensor with exact shape/dtype: initializer features'):
                raise
            raise old.FitFailure(str(exc)) from exc

    def condition(self, q, u):
        return self._numeric_call(super().condition, q, u)

    def forward(self, future, state):
        return self._numeric_call(super().forward, future, state)


def model_for(arm, seed, linear):
    old.require(arm in ALL_ARMS, 'declared history family required')
    if arm in ARMS:
        return HistoryAdapter(seed, arm)
    return transition.model_for('gru_residual', seed, linear) if arm == 'gru10' else structured.model_for(arm, seed, linear)


def resource_model(model, arm):
    if arm == 'gru10':
        return transition.resource_model(model, 'gru_residual')
    return structured.resource_model(model, arm)


def initial_pairing(model):
    digest = hashlib.sha256()
    for name, value in model.cell.state_dict().items():
        array = value.detach().cpu().numpy()
        digest.update(name.encode() + b'\0' + str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())
    zero = model.initializer == 'last_two' or bool(torch.count_nonzero(model.head.weight) == 0
                                                  and torch.count_nonzero(model.head.bias) == 0)
    old.require(zero, 'zero initializer head required')
    return {'common_cell_sha256': digest.hexdigest(), 'zero_head': zero,
            'common_parameters': 590, 'added_parameters': model.added_parameter_count}


def check_deadline(clock, started_ns, seconds, *, whole):
    if clock.now_ns() - started_ns >= int(seconds * 1_000_000_000):
        error = old.WholeStudyTimeout if whole else old.FitFailure
        raise error('native whole-study deadline exceeded' if whole else 'native single-fit deadline exceeded')


def effective_status(fit):
    return fit.get('effective_status', fit['fit']['status'])


def select(rows, cfg):
    options, selected = {}, {}
    expected = {(name, seed) for name in cfg['partitions']['dev'] for seed in cfg['seeds']}
    for arm in ALL_ARMS:
        options[arm] = []
        for rate in cfg['learning_rates']:
            subset = [r for r in rows if r['arm'] == arm and r['learning_rate'] == rate and r['horizon'] == 128]
            valid = (len(subset) == len(expected) and {(r['recording'], r['seed']) for r in subset} == expected
                     and all(old.valid_metric_row(r) for r in subset))
            score = float(np.sqrt(sum(r['metrics']['standardized_sse'] for r in subset)
                                  / sum(r['metrics']['scalars'] for r in subset))) if valid else None
            options[arm].append({'rate': rate, 'eligible': valid, 'pooled_rmse': score})
        eligible = [o for o in options[arm] if o['eligible']]
        selected[arm] = min(eligible, key=lambda o: (o['pooled_rmse'], o['rate']))['rate'] if eligible else None
    return {'selected_rates': selected, 'options': options,
            'scope': 'pooled H128 exposed DEV2/all3seeds, lower-rate tie; cached full two-rate recipes replayed without fitting'}


def selected_cost(arm, selection, resources, cfg):
    subset = [r for r in resources if r['arm'] == arm]
    expected = set(cfg['seeds']) if arm in ALL_ARMS else {None}
    if len(subset) != len(expected) or {r['seed'] for r in subset} != expected:
        return None
    values, sizes = [], []
    for row in subset:
        if arm in ALL_ARMS and row.get('learning_rate') != selection['selected_rates'][arm]:
            return None
        value = row['timing']['median_seconds']
        parts = [row[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
        if not (type(value) in (int, float) and math.isfinite(value) and value > 0
                and all(type(v) is int and v >= 0 for v in parts)):
            return None
        values.append(value); sizes.append(sum(parts))
    return {'latency': float(np.median(values)), 'bytes': max(sizes)}


def evaluate_rule(rows, selection, resources, cfg):
    old.require(selection == select(rows, cfg), 'selection must match full fixed recipe rows')
    rates = selection['selected_rates']
    details, equal_means, costs = {}, {}, {}
    for recording in cfg['partitions']['dev']:
        per_arm, paired = {}, {}
        for arm in (*ALL_ARMS, *REFERENCES):
            subset = [r for r in rows if r['recording'] == recording and r['horizon'] == 128 and r['arm'] == arm
                      and (arm in REFERENCES or r['learning_rate'] == rates[arm])]
            expected = set(cfg['seeds']) if arm in ALL_ARMS else {None}
            valid = (len(subset) == len(expected) and {r['seed'] for r in subset} == expected
                     and all(old.valid_metric_row(r) for r in subset))
            if valid:
                per_arm[arm] = float(np.mean([r['metrics']['standardized_rmse'] for r in subset]))
                paired[arm] = {r['seed']: r['metrics']['standardized_rmse'] for r in subset}
        differences = {other: {str(seed): paired['temporal_affine'][seed] - paired[other][seed]
                              for seed in cfg['seeds']}
                       for other in ARMS[:2] if 'temporal_affine' in paired and other in paired}
        details[recording] = {'means': per_arm, 'paired_primary_differences': differences}
    for arm in (*ALL_ARMS, *REFERENCES):
        if all(arm in d['means'] for d in details.values()):
            equal_means[arm] = float(np.mean([d['means'][arm] for d in details.values()]))
        costs[arm] = selected_cost(arm, selection, resources, cfg)
    candidate = equal_means.get('temporal_affine')
    complete_primary = all(rates[a] is not None and a in equal_means for a in ARMS)
    improvement = (candidate is not None and all(equal_means.get(a, 0) > 0
                   and candidate <= (1-cfg['mean_reduction']) * equal_means[a] for a in ARMS[:2]))
    no_harm = all(all(a in d['means'] for a in ARMS) and d['means']['temporal_affine'] <= cfg['file_harm_ratio']
                  * min(d['means'][a] for a in ARMS[:2]) for d in details.values())
    left, baseline = costs['temporal_affine'], costs['last_two']
    latency = left is not None and baseline is not None and left['latency'] <= cfg['latency_ratio'] * baseline['latency']
    complete_frontier = all(a in equal_means and costs[a] is not None for a in (*ALL_ARMS, *REFERENCES))
    dominators = []
    if complete_frontier:
        for arm in (*ALL_ARMS, *REFERENCES):
            if arm == 'temporal_affine':
                continue
            control = (equal_means[arm], costs[arm]['latency'], costs[arm]['bytes'])
            primary = (candidate, left['latency'], left['bytes'])
            if all(a <= b for a, b in zip(control, primary, strict=True)) and any(a < b for a, b in zip(control, primary, strict=True)):
                dominators.append(arm)
    conditions = [{'name': name, 'passed': bool(value)} for name, value in (
        ('primary_recipes_complete', complete_primary), ('equal_file_mean_5pct_vs_both_locals', improvement),
        ('each_file_within_2pct_best_local', no_harm), ('latency_within_125pct_last_two', latency),
        ('complete_frontier_not_dominated', complete_frontier and not dominators))]
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if passed == 5 else 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION',
            'passed': passed, 'total': 5, 'conditions': conditions, 'details': details,
            'equal_file_means': equal_means, 'costs': costs, 'frontier_complete': complete_frontier,
            'dominators': dominators, 'scope': 'exposed-DEV information comparison; no novelty, independent confirmation or control claim'}


def permuted_batch(batch):
    return {**batch, 'q_context': batch['q_context'][:, PERMUTATION, :].copy(),
            'u_context': batch['u_context'][:, PERMUTATION, :].copy()}


def check_permutation(arm, original, changed):
    old.require(original.shape == changed.shape and np.isfinite(original).all() and np.isfinite(changed).all(),
                'finite matching diagnostic predictions required')
    exact = bool(np.array_equal(original, changed))
    if arm in ARMS[:2]:
        old.require(exact, 'local permutation invariance failed; diagnostic validity error')
    return {'exact_invariance': exact, 'maximum_absolute_change': float(np.max(np.abs(original - changed))),
            'scope': 'fixed older-order corruption diagnostic only; not a sixth scientific gate'}


def diagnostic_prediction(model, arm, corrupted, original):
    """OOD numerical failure is retained without changing the five science gates."""
    old.require(arm in ARMS and np.isfinite(original).all(), 'finite original primary diagnostic required')
    prediction, error = None, None
    try:
        with torch.no_grad():
            prediction = old.infer(model, corrupted).numpy().astype(np.float64)
    except old.FitFailure as exc:
        error = {'type': 'NonfiniteDiagnostic', 'message': str(exc)}
    if prediction is not None:
        old.require(prediction.shape == original.shape, 'diagnostic prediction shape changed')
        if not np.isfinite(prediction).all():
            error = {'type': 'NonfiniteDiagnostic', 'message': 'nonfinite reversed-history prediction'}
    if error is not None:
        old.require(arm == 'temporal_affine', 'local permutation inference failed; diagnostic validity error')
        return {'prediction': prediction, 'status': 'FAILED', 'error': error, 'check': None}
    return {'prediction': prediction, 'status': 'PASS', 'error': None,
            'check': check_permutation(arm, original, prediction)}


def cached_fit_records(plan, output, models, linear):
    records = []
    for group in PARENT_FOLDERS:
        for filename in ('fits.json', 'results.json'):
            copy_parent(plan, group, filename, output / f'parent-{group}-{filename}')
        originals = json.loads((output / f'parent-{group}-fits.json').read_text())
        wanted = CACHED_ARMS[:-1] if group == 'structured' else ('gru_residual',)
        expected = {f'{a}-{s}-lr{ri}' for a in wanted for s in old.SEEDS for ri in range(2)}
        subset = [f for f in originals if f['arm'] in wanted]
        old.require(len(subset) == len(expected) and {f['key'] for f in subset} == expected, 'complete cached recipe roster')
        for original in subset:
            source_key, source_arm, seed, rate = (original[k] for k in ('key', 'arm', 'seed', 'learning_rate'))
            arm = 'gru10' if group == 'transition' else source_arm
            ri = config()['learning_rates'].index(rate)
            old.require(source_key == f'{source_arm}-{seed}-lr{ri}' and seed in old.SEEDS, 'cached identity')
            key = f'{arm}-{seed}-lr{ri}'
            for filename in LEGACY_FILES:
                copy_parent(plan, group, source_key + '/' + filename, output / key / filename)
            old.require(json.loads((output / key / 'fit-receipt.json').read_text()) == original['fit'], 'original fit receipt join')
            if effective_status(original) == 'PASS':
                model = model_for(arm, seed, linear)
                state = load_parent_arrays(plan, group, source_key + '/final.npz')
                model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
                model.zero_grad(set_to_none=True)
                old.require(resource_model(model, arm) == original['resources'], 'unchanged cached resource counts')
                models[key] = model
            records.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate,
                            'fit': original['fit'], 'effective_status': effective_status(original),
                            'resources': original['resources'], 'origin': 'cached_parent', 'parent_group': group,
                            'parent_key': source_key, 'parent_fit': original,
                            'parent_files': {name: plan['parent_payloads'][group][source_key + '/' + name] for name in LEGACY_FILES}})
    old.require(len(records) == 30, 'all30 cached attempts required')
    return records


def run(registration, output):
    clock = SuspendClock()
    started_ns, started = clock.now_ns(), time.monotonic()
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
                       'thread_env': {k: os.environ.get(k) for k in old.THREADS}, 'torch_threads': torch.get_num_threads(),
                       'clock': clock.backend, 'timing_scope': 'complete physical request plus uniform native deadline callback'})
        with (output / 'registration.json').open('xb') as handle:
            handle.write(Path(registration).read_bytes())
        for name in SOURCES:
            path = output / 'sources' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            old.require(old.descriptor(path) == plan['sources'][name], 'source snapshot changed')
        physical_fit = [{'name': name, **load_arrays(plan, 'fit-data-' + name + '.npz')} for name in cfg['partitions']['fit']]
        norm = old.normalizers(physical_fit, cfg['skip'])
        original_norm = load_arrays(plan, 'normalizers.npz')
        parent_norm = load_parent_arrays(plan, 'structured', 'normalizers.npz')
        old.require(set(norm) == set(original_norm) == set(parent_norm)
                    and all(np.array_equal(norm[k], original_norm[k]) and np.array_equal(norm[k], parent_norm[k]) for k in norm),
                    'unchanged FIT-only normalization required')
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        linear = load_parent_arrays(plan, 'structured', 'linear.npz')['coefficient']
        for filename in ('normalizers.npz', 'linear.npz'):
            copy_parent(plan, 'structured', filename, output / filename)
        ridges = {}
        for penalty in cfg['direct_ridges']:
            arm = f'causal_ridge_{int(penalty)}'
            saved = load_parent_arrays(plan, 'structured', arm + '.npz')
            item = plan['parent_payloads']['structured'][arm + '.json']
            _pin(item)
            metadata = json.loads(Path(item['path']).read_text())
            old.require(set(saved) == {f'h{h:03d}' for h in range(1, 129)}, 'complete cached causal bank')
            ridges[arm] = CausalRobotRidge(tuple(saved[f'h{h:03d}'] for h in range(1, 129)), penalty, metadata['fit_rows'])
            old.require(ridges[arm].metadata() == {k: v for k, v in metadata.items() if k != 'fit_seconds'}, 'cached ridge metadata')
            for suffix in ('.npz', '.json'):
                copy_parent(plan, 'structured', arm + suffix, output / (arm + suffix))
            check()
        initial_cells = {}
        for seed in cfg['seeds']:
            batches = load_parent_arrays(plan, 'structured', f'batches-{seed}.npz')
            expected_batches = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(batches) == set(expected_batches) and all(np.array_equal(batches[k], expected_batches[k]) for k in batches),
                        'exact parent paired batches required')
            copy_parent(plan, 'structured', f'batches-{seed}.npz', output / f'batches-{seed}.npz')
            for ri, rate in enumerate(cfg['learning_rates']):
                for arm in ARMS:
                    check()
                    key = f'{arm}-{seed}-lr{ri}'
                    before, fit_started_ns = time.monotonic(), clock.now_ns()
                    def fit_check(native_start=fit_started_ns):
                        check()
                        check_deadline(clock, native_start, cfg['fit_cap_seconds'], whole=False)
                    model = model_for(arm, seed, linear)
                    pairing = initial_pairing(model)
                    initial_cells.setdefault(seed, pairing['common_cell_sha256'])
                    old.require(initial_cells[seed] == pairing['common_cell_sha256'], 'paired common initialization changed')
                    fit = old.train_one(model, fit_data, batches, cfg=cfg, lr=rate, folder=output / key,
                                        check=fit_check, fit_started=before)
                    model.zero_grad(set_to_none=True)
                    native_seconds = (clock.now_ns() - fit_started_ns) / 1e9
                    effective = 'FAILED' if native_seconds >= cfg['fit_cap_seconds'] else fit['status']
                    models[key] = model
                    fits.append({'key': key, 'arm': arm, 'seed': seed, 'learning_rate': rate, 'fit': fit,
                                 'effective_status': effective, 'native_fit_seconds': native_seconds,
                                 'native_fit_cap_seconds': cfg['fit_cap_seconds'],
                                 'native_fit_error': {'type': 'FitFailure', 'message': 'native fit cap crossed including preservation'}
                                 if native_seconds >= cfg['fit_cap_seconds'] else None,
                                 'initial_pairing': pairing, 'resources': resource_model(model, arm), 'origin': 'fresh'})
                    old.write_json(output / f'completed-fit-{len(fits):02d}.json', fits[-1])
                    check()
                    print(json.dumps({'completed': len(fits), 'key': key, 'status': effective,
                                      'helper_status': fit['status'], 'updates': fit['completed_updates']}), flush=True)
        old.require(len(fits) == 18, 'all18 fresh attempts before current DEV load')
        for record in cached_fit_records(plan, output, models, linear):
            fits.append(record)
            old.write_json(output / f'completed-fit-{len(fits):02d}.json', record)
            check()
        old.require(len(fits) == 48, 'complete48 fit roster required')
        old.write_json(output / 'checkpoint-barrier.json', {'fresh_fit_attempts': 18, 'cached_fit_records': 30,
                       'fit_records': 48, 'dev_loads_this_run': 0, 'dev_exposed_prior': True,
                       'checkpoints': {f['key']: old.descriptor(output / f['key'] / 'final.npz') for f in fits}})
        physical_timing, dev_batches, ordinary = None, {}, {}
        for name in cfg['partitions']['dev']:
            physical = {'name': name, **load_arrays(plan, 'dev-data-' + name + '.npz')}
            data = old.normalized_record(physical, norm)
            starts = old.dev_windows(len(data['q']), cfg)
            batch = old.window_batch([data], {'record': np.zeros(len(starts), dtype=np.int64), 'start': starts}, 32, 128)
            dev_batches[name] = batch
            np.savez_compressed(output / ('dev-windows-' + name + '.npz'), starts=starts, target=batch['target'])
            if physical_timing is None:
                physical_timing = old.window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, 32, 128)
            for fit in fits:
                prediction, error = None, None
                check()
                if effective_status(fit) == 'PASS':
                    try:
                        with torch.no_grad():
                            prediction = old.infer(models[fit['key']], batch).numpy().astype(np.float64)
                    except old.FitFailure as exc:
                        error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                    if prediction is not None:
                        np.savez_compressed(output / ('prediction-' + name + '-' + fit['key'] + '.npz'), prediction=prediction)
                        if fit['arm'] in ARMS:
                            ordinary[(name, fit['key'])] = prediction
                else:
                    error = {'type': 'FailedTrainingAttempt', 'effective_status': effective_status(fit)}
                common = {'recording': name, 'arm': fit['arm'], 'seed': fit['seed'],
                          'learning_rate': fit['learning_rate'], 'fit_key': fit['key']}
                rows.extend(old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error))
            for arm in REFERENCES:
                prediction = predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges else old.reference_predict(arm, {'linear_frozen': linear}, batch)
                np.savez_compressed(output / ('prediction-' + name + '-' + arm + '.npz'), prediction=prediction)
                rows.extend(old.scored_rows({'recording': name, 'arm': arm, 'seed': None, 'learning_rate': None},
                                            prediction, batch['target'], norm['q_std'], cfg))
            check()
        old.require(len(rows) == 208, 'all208 ordinary metric rows required')
        selection = select(rows, cfg)
        diagnostic_rows, diagnostic_checks = [], []
        for name, batch in dev_batches.items():
            corrupted = permuted_batch(batch)
            for arm in ARMS:
                rate = selection['selected_rates'][arm]
                for seed in cfg['seeds']:
                    check()
                    common = {'recording': name, 'arm': arm, 'seed': seed, 'learning_rate': rate}
                    if rate is None:
                        diagnostic_checks.append({**common, 'status': 'UNAVAILABLE', 'reason': 'no complete selected primary recipe'})
                        diagnostic_rows.extend(old.scored_rows(common, None, batch['target'], norm['q_std'], cfg,
                                                               {'type': 'UnavailableSelectedRecipe'}))
                        continue
                    key = f'{arm}-{seed}-lr{cfg["learning_rates"].index(rate)}'
                    diagnostic = diagnostic_prediction(models[key], arm, corrupted, ordinary[(name, key)])
                    prediction = diagnostic['prediction']
                    if prediction is not None:
                        np.savez_compressed(output / ('permuted-prediction-' + name + '-' + key + '.npz'), prediction=prediction)
                    diagnostic_checks.append({**common, 'status': diagnostic['status'], 'error': diagnostic['error'],
                                              'check': diagnostic['check'], 'prediction_saved': prediction is not None})
                    diagnostic_rows.extend(old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg,
                                                           diagnostic['error']))
        old.write_json(output / 'permutation.json', {'permutation': list(PERMUTATION), 'checks': diagnostic_checks,
                       'rows': diagnostic_rows, 'scientific_gate': False,
                       'scope': 'paired older-order corruption; local exact invariance is a diagnostic validity condition'})
        for fit in fits:
            if selection['selected_rates'][fit['arm']] == fit['learning_rate']:
                old.require(all(p.grad is None for p in models[fit['key']].parameters()), 'no retained inference gradients')
                def pred(batch, key=fit['key']):
                    check()
                    return old.infer(models[key], batch).numpy().astype(np.float64)
                resources.append({'arm': fit['arm'], 'seed': fit['seed'], 'learning_rate': fit['learning_rate'],
                                  **fit['resources'], 'timing': timed_request(pred, physical_timing, norm, cfg)})
        for arm in REFERENCES:
            reference_rows = [r for r in rows if r['arm'] == arm and r['horizon'] == 128]
            if len(reference_rows) != 2 or not all(old.valid_metric_row(r) for r in reference_rows):
                # Missing/nonfinite comparison costs must fail the complete
                # frontier, never silently remove a possible dominator.
                continue
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
        old.write_json(output / 'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p)
                       for p in sorted(output.rglob('*')) if p.is_file()}})
        check()
        old.write_json(output / 'receipt.json', {'status': 'PASS', 'registration_sha256': sha,
                       'seconds': (clock.now_ns() - started_ns) / 1e9, 'clock': clock.backend,
                       'monotonic_seconds': time.monotonic() - started, 'fits': len(fits),
                       'fresh_fit_attempts': 18, 'cached_fit_records': 30, 'rows': len(rows),
                       'permutation_metric_rows': len(diagnostic_rows), 'scientific_result': result['status'],
                       'raw_decodes': 0, 'reference_refits': 0, 'parent_refits': 0,
                       'saved_fit_loads': 7, 'saved_dev_loads': 2, 'confirmation_access': False, 'official_test_access': False})
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
