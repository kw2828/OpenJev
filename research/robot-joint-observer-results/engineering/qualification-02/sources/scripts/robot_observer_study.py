"""Frozen dynamics, learned prefix correction, exposed development only."""
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
import publish_robot_history_confirmation as publication
import robot_history_confirmation as confirmation
import robot_history_initialization_study as history
import torch

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.robot_observer_initializer import FrozenObserverInitializer
from openjev.research.suspend_clock import SuspendClock

old = history.old
ROOT = old.ROOT
VERSION = 'robot-observer-study-v1'
LEARNED = ('local_affine', 'temporal_affine', 'observer_learned')
FIXED = ('last_two', 'observer_fixed', 'observer_zero')
CACHED = ('joint_local_affine', 'joint_temporal_affine', 'dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
REFERENCES = history.REFERENCES
FAMILIES = (*LEARNED, *FIXED, *CACHED, *REFERENCES)
SIMPLE = ('last_two', 'local_affine', 'temporal_affine', 'observer_fixed')
CACHED_ORIGINAL = {a: a.removeprefix('joint_') for a in CACHED}
CACHED_RATES = {a: confirmation.FIXED_RATES[CACHED_ORIGINAL[a]] for a in CACHED}
DEV = ('recording_2021_12_15_21H_54M.mat', 'recording_2021_12_15_22H_10M.mat')
EXPOSED = (*DEV, *confirmation.CONFIRM)
HISTORY = ROOT / 'output/robot-history-initialization-study-v1'
CONFIRMATION = ROOT / 'output/robot-history-confirmation-v1'
SOURCES = (*confirmation.SOURCES, 'scripts/publish_robot_history_confirmation.py',
           'tests/test_publish_robot_history_confirmation.py', 'src/openjev/research/robot_observer_initializer.py',
           'tests/test_robot_observer_initializer.py', 'scripts/robot_observer_study.py',
           'tests/test_robot_observer_study.py', 'research/robot-observer-protocol.md',
           'scripts/audit_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py')
LAUNCHER = 'scripts/launch_robot_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:], LAUNCHER)
CONDITIONS = ('complete_forecasts_and_costs', 'equal_four_file_mean_5pct_vs_best_control',
              'each_file_within_2pct_best_simple', 'latency_within_150pct_last_two', 'complete_frontier_not_dominated')
NUMERIC_ERRORS = (*confirmation.NUMERIC_ERRORS, 'finite CPU tensor with exact shape/dtype: observer innovation')


def config():
    parent = history.config()
    keys = ('context', 'train_horizon', 'dev_horizon', 'horizons', 'skip', 'dev_stride', 'updates', 'batch_size',
            'adam_betas', 'adam_eps', 'gradient_clip', 'window_seed_offset', 'preprocessing_sha256',
            'timing_warmups', 'timing_repeats', 'seeds', 'learning_rates')
    return {**{k: parent[k] for k in keys}, 'version': VERSION, 'learned': list(LEARNED), 'fixed': list(FIXED),
            'cached': list(CACHED), 'references': list(REFERENCES), 'families': list(FAMILIES),
            'cached_rates': CACHED_RATES, 'partitions': {'fit': parent['partitions']['fit'], 'dev': list(DEV), 'exposed': list(EXPOSED)},
            'fit_cap_seconds': 1800., 'wall_cap_seconds': 14400., 'mean_reduction': .05, 'file_harm_ratio': 1.02,
            'latency_ratio': 1.5, 'all_evaluation_data_exposed': True, 'official_test_access': False,
            'raw_mat_access': False, 'normalizer_refits': 0, 'reference_refits': 0,
            'selection_scope': 'pooled H128 SSE on original two DEV files only; lower rate wins exact ties'}


def identities():
    rows = []
    for seed in old.SEEDS:
        for ri, rate in enumerate((.001, .003)):
            rows += [{'key': f'{a}-{seed}-lr{ri}', 'arm': a, 'seed': seed, 'learning_rate': rate, 'origin': 'fresh'} for a in LEARNED]
    rows += [{'key': f'{a}-{s}-fixed', 'arm': a, 'seed': s, 'learning_rate': None, 'origin': 'fixed'} for a in FIXED for s in old.SEEDS]
    rows += [{'key': f'{a}-{s}-cached', 'arm': a, 'seed': s, 'learning_rate': CACHED_RATES[a], 'origin': 'cached'} for a in CACHED for s in old.SEEDS]
    rows += [{'key': a, 'arm': a, 'seed': None, 'learning_rate': None, 'origin': 'reference'} for a in REFERENCES]
    return rows


def pin(path):
    return {'path': str(Path(path).resolve()), **old.descriptor(path)}


def read(path):
    return publication.read(path)


def expected_inputs(prior):
    """Opaque lineage resolution; no numerical decoder is used."""
    registered = ROOT / 'research/robot-history-initialization-registration.json'
    old.require(old.descriptor(registered)['sha256'] == confirmation.PARENT_SHA, 'exact historical registration')
    previous = read(registered)
    result = {'parent/' + n: item for n, item in prior['plan']['parent_payloads'].items()}
    inventory = read(HISTORY / 'manifest.json')['files']
    for n in ('fits.json', *(f'batches-{s}.npz' for s in old.SEEDS)):
        item = pin(HISTORY / n)
        old.require({k: item[k] for k in ('sha256', 'bytes')} == inventory[n], 'historical original payload')
        result['parent/' + n] = item
    for part in ('fit', 'dev'):
        for name in previous['config']['partitions'][part]:
            result['data/' + name] = previous['data'][part + '-data-' + name + '.npz']
    for name in confirmation.CONFIRM:
        n = 'confirm-data-' + name + '.npz'
        item = pin(CONFIRMATION / n)
        old.require({k: item[k] for k in ('sha256', 'bytes')} == prior['inventory'][n], 'audited exposed recording')
        result['data/' + name] = item
    old.require(len(result) == 45, 'exact45 inherited payloads')
    return result


def authenticate(registration):
    raw = Path(registration).read_bytes(); plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact observer config')
    old.require(set(plan['sources']) == set(SOURCES) and len(SOURCES) == 45, 'all45 inherited and new sources')
    for name, item in plan['sources'].items():
        old.require(old.descriptor(ROOT/name) == item, 'source drift: ' + name)
    old.require(plan['launcher'] == old.descriptor(ROOT/LAUNCHER), 'launcher drift')
    old.require(all(os.environ.get(k) == '1' for k in old.THREADS), 'single-thread environment')
    confirmation._pin(plan['qualification']); q = read(plan['qualification']['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_observer_initializer.py',
                 'tests/test_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py']]
    old.require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
                and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(old.THREADS, '1')
                and [r['command'] for r in q['commands']] == commands, 'exact prefit qualification')
    for row in q['commands']:
        old.require(row['returncode'] == 0 and old.descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification log')
    prior = publication.authenticate(ROOT/publication.STUDY, ROOT/publication.AUDIT, ROOT/publication.ENGINEERING)
    old.require(plan['inputs'] == expected_inputs(prior), 'exact audited parent/data inputs; TEST unsupported')
    for item in plan['inputs'].values():
        confirmation._pin(item)
    old.require(plan['parent_publication_manifest'] == pin(ROOT/publication.OUTPUT/'manifest.json')
                and plan['parent_audit'] == pin(ROOT/publication.AUDIT), 'published failed confirmation lineage')
    return plan, hashlib.sha256(raw).hexdigest()


def load(plan, name):
    old.require(name in plan['inputs'] and name in expected_input_names(), 'registered numerical input only')
    item = plan['inputs'][name]; confirmation._pin(item)
    old.require(item['path'].endswith('.npz'), 'only NPZ inputs may enter numerical decoder')
    with np.load(item['path'], allow_pickle=False) as values:
        arrays = {k: values[k].copy(order='K') for k in values.files}
    if name.startswith('data/'):
        old.require(set(arrays) == {'q', 'u', 'raw_indices'} and all(arrays[k].dtype == np.float64
                    and arrays[k].shape == (3636, 6) and np.isfinite(arrays[k]).all() for k in ('q', 'u'))
                    and arrays['raw_indices'].dtype == np.int64
                    and np.array_equal(arrays['raw_indices'], np.arange(0, 90881, 25)), 'complete processed recording schema')
    return arrays


def expected_input_names():
    return {'parent/' + n for n in (*confirmation.PAYLOADS, 'fits.json', *(f'batches-{s}.npz' for s in old.SEEDS))} | {
        'data/' + n for n in (*config()['partitions']['fit'], *EXPOSED)}


class ObserverAdapter(FrozenObserverInitializer):
    @staticmethod
    def numeric(function, *args):
        try:
            return function(*args)
        except ValueError as exc:
            if str(exc) not in NUMERIC_ERRORS:
                raise
            raise old.FitFailure(str(exc)) from exc

    def condition(self, q, u):
        return self.numeric(super().condition, q, u)

    def forward(self, u, state):
        return self.numeric(super().forward, u, state)


def model_for(arm, seed, linear=None):
    old.require(arm in (*LEARNED, *FIXED, *CACHED), 'declared neural model')
    return ObserverAdapter(seed, arm) if arm in (*LEARNED, *FIXED) else history.model_for(CACHED_ORIGINAL[arm], seed, linear)


def cell_hash(model):
    return confirmation.parameter_hash(model.cell)


def check_frozen(model, expected_sha):
    old.require(cell_hash(model) == expected_sha and all(not p.requires_grad and p.grad is None for p in model.cell.parameters()),
                'frozen590 transition changed or received gradients')


def check_optimizer(model, arrays, completed_updates):
    names = {n: p for n, p in model.named_parameters() if p.requires_grad}
    old.require(set(names) in ({'gain'}, {'head.weight', 'head.bias'}), 'initializer-only trainable ownership')
    old.require(set(arrays) == {n+'/'+s for n in names for s in ('step', 'exp_avg', 'exp_avg_sq')}, 'initializer-only Adam state')
    for n, p in names.items():
        step = arrays[n+'/step']
        old.require(step.shape == () and np.isfinite(step).all() and float(step) == completed_updates, 'exact Adam update count')
        for s in ('exp_avg', 'exp_avg_sq'):
            a = arrays[n+'/'+s]
            old.require(a.shape == tuple(p.shape) and a.dtype == np.float32 and np.isfinite(a).all(), 'finite Adam moment schema')


def resource_model(model, arm):
    if arm in CACHED:
        return history.resource_model(model, CACHED_ORIGINAL[arm])
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'buffer_bytes': sum(p.numel()*p.element_size() for p in model.buffers()),
            'state_scalars': 12, 'state_bytes': 48, 'normalizer_bytes': 192, 'inactive_parameters': 0,
            'dtype': 'float32', 'input_bytes': 9216, 'output_bytes': 6144, 'temporary_workspace': 'not measured'}


def selected(row, selection):
    return row['arm'] not in LEARNED or row['learning_rate'] == selection['selected_rates'][row['arm']]


def resource_identities(selection):
    rows = [r for r in identities() if selected(r, selection)]
    for arm in LEARNED:
        if selection['selected_rates'][arm] is None:
            rows.extend({'key': f'{arm}-{s}-unavailable', 'arm': arm, 'seed': s, 'learning_rate': None,
                         'origin': 'unavailable'} for s in old.SEEDS)
    old.require(len(rows) == 43, 'all43 planned resource identities')
    return rows


def validate_rows(rows, cfg):
    expected = {(n, i['key'], h) for n in EXPOSED for i in identities() for h in (64, 128)}
    ids = [(r['recording'], r['fit_key'], r['horizon']) for r in rows]
    roster = {r['key']: r for r in identities()}
    old.require(len(ids) == len(set(ids)) == 416 and set(ids) == expected, 'all416 scheduled score rows')
    for r in rows:
        old.require(all(r[k] == roster[r['fit_key']][k] for k in ('arm', 'seed', 'learning_rate')), 'score identity joins')
        old.require(r['status'] in ('PASS', 'FAILED'), 'declared score status')
        if r['status'] == 'PASS':
            old.require(r['error'] is None and r['metrics']['windows'] == 22
                        and r['metrics']['scalars'] == 22*r['horizon']*6 and r['metrics']['horizon'] == r['horizon'],
                        'exact complete metric geometry')
        else:
            old.require(r['error'] is not None and r['metrics'] is None, 'explicit failed metric')
    old.require(cfg == config(), 'immutable observer config')


def select(rows, cfg):
    validate_rows(rows, cfg)
    choices, options = {}, {}
    for arm in LEARNED:
        scores = []
        for rate in cfg['learning_rates']:
            subset = [r for r in rows if r['arm'] == arm and r['learning_rate'] == rate and r['horizon'] == 128 and r['recording'] in DEV]
            complete = len(subset) == 6 and all(old.valid_metric_row(r) for r in subset)
            score = sum(r['metrics']['standardized_sse'] for r in subset) if complete else None
            scores.append({'learning_rate': rate, 'complete': complete, 'pooled_sse': score})
        available = [r for r in scores if r['complete']]
        choices[arm] = min(available, key=lambda r: (r['pooled_sse'], r['learning_rate']))['learning_rate'] if available else None
        options[arm] = scores
    return {'selected_rates': choices, 'options': options, 'scope': cfg['selection_scope']}


def evaluate_rule(rows, selection, resources, cfg):
    validate_rows(rows, cfg)
    old.require(selection == select(rows, cfg), 'DEV2-only selection identity')
    details, means, costs = {}, {}, {}
    for name in EXPOSED:
        values = {}
        for arm in FAMILIES:
            subset = [r for r in rows if r['recording'] == name and r['arm'] == arm and r['horizon'] == 128 and selected(r, selection)]
            count = 1 if arm in REFERENCES else 3
            if len(subset) == count and all(old.valid_metric_row(r) for r in subset):
                values[arm] = float(np.mean([r['metrics']['standardized_rmse'] for r in subset]))
        details[name] = {'means': values}
    expected = {(r['key'], r['arm'], r['seed'], r['learning_rate']) for r in resource_identities(selection)}
    actual = [(r['key'], r['arm'], r['seed'], r['learning_rate']) for r in resources]
    old.require(len(actual) == len(set(actual)) and set(actual) == expected, 'all selected timing attempts, no omissions')
    for arm in FAMILIES:
        if all(arm in d['means'] for d in details.values()):
            means[arm] = float(np.mean([d['means'][arm] for d in details.values()]))
        subset = [r for r in resources if r['arm'] == arm]
        good = len(subset) == (1 if arm in REFERENCES else 3)
        times, sizes = [], []
        for r in subset:
            timing = r.get('timing'); duration = timing.get('median_seconds') if isinstance(timing, dict) else None
            parts = [r.get(k) for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')]
            good = good and r.get('status') == 'PASS' and r.get('error') is None
            good = good and type(duration) in (float, int) and math.isfinite(duration) and duration > 0
            good = good and all(type(v) is int and v >= 0 for v in parts)
            if good:
                times.append(duration); sizes.append(sum(parts))
        costs[arm] = {'latency': float(np.median(times)), 'bytes': max(sizes)} if good else None
    frontier = all(a in means and costs[a] is not None for a in FAMILIES)
    complete = frontier and all(selection['selected_rates'][a] is not None for a in LEARNED)
    complete = complete and all(old.valid_metric_row(r) for r in rows if selected(r, selection))
    candidate = means.get('observer_learned'); controls = [means.get(a) for a in FAMILIES if a != 'observer_learned']
    best = min(controls) if all(v is not None for v in controls) else None
    improvement = candidate is not None and best is not None and best > 0 and candidate <= .95*best
    no_harm = all(all(a in d['means'] for a in (*SIMPLE, 'observer_learned')) and d['means']['observer_learned']
                  <= 1.02*min(d['means'][a] for a in SIMPLE) for d in details.values())
    left, base = costs['observer_learned'], costs['last_two']
    latency = left is not None and base is not None and left['latency'] <= 1.5*base['latency']
    dominators = []
    if frontier:
        point = (candidate, left['latency'], left['bytes'])
        for a in FAMILIES:
            if a == 'observer_learned': continue
            other = (means[a], costs[a]['latency'], costs[a]['bytes'])
            if all(x <= y for x, y in zip(other, point, strict=True)) and any(x < y for x, y in zip(other, point, strict=True)):
                dominators.append(a)
    conditions = [{'name': n, 'passed': bool(v)} for n, v in zip(CONDITIONS, (complete, improvement, no_harm, latency, frontier and not dominators), strict=True)]
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'OBSERVER_DEVELOPMENT_PASS' if passed == 5 else 'OBSERVER_DEVELOPMENT_FAIL', 'passed': passed, 'total': 5,
            'conditions': conditions, 'details': details, 'equal_file_means': means, 'costs': costs,
            'best_control_mean': best, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'All four files exposed development; not confirmation, independent robots, novelty or control'}


def run(registration, output):
    clock = SuspendClock(); start_ns, started = clock.now_ns(), time.monotonic()
    plan, sha = authenticate(registration); cfg = config(); output = Path(output)
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    fits, models, rows, resources, states = [], {}, [], [], {}
    def check(): history.check_deadline(clock, start_ns, cfg['wall_cap_seconds'], whole=True)
    def copy_input(name):
        item = plan['inputs'][name]; confirmation._pin(item); dst = output/name; dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('xb') as f: f.write(Path(item['path']).read_bytes())
        old.require(old.descriptor(dst) == {k: item[k] for k in ('sha256', 'bytes')}, 'exact inherited copy')
    try:
        old.write_json(output/'runtime.json', {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'machine': platform.machine(), 'thread_env': {k: os.environ.get(k) for k in old.THREADS},
            'torch_threads': torch.get_num_threads(), 'clock': clock.backend})
        (output/'registration.json').write_bytes(Path(registration).read_bytes())
        for name in SOURCES:
            dst = output/'sources'/name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes((ROOT/name).read_bytes())
            old.require(old.descriptor(dst) == plan['sources'][name], 'source snapshot')
        for name in plan['inputs']:
            if name.startswith('parent/'): copy_input(name)
        norm = load(plan, 'parent/normalizers.npz'); linear = load(plan, 'parent/linear.npz')['coefficient']
        physical_fit = [{'name': n, **load(plan, 'data/'+n)} for n in cfg['partitions']['fit']]
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        batches = {s: load(plan, f'parent/batches-{s}.npz') for s in old.SEEDS}
        for seed, bank in batches.items():
            expected = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(bank) == set(expected) and all(np.array_equal(bank[k], expected[k]) for k in bank), 'same paired4096 batches')
        cells = {s: load(plan, f'parent/last_two-{s}-lr0/final.npz') for s in old.SEEDS}
        def initialize(arm, seed):
            model = model_for(arm, seed, linear); arrays = cells[seed]
            old.require(set(arrays) == {'cell.'+n for n in model.cell.state_dict()}, 'selected seed exact cell schema')
            old.require(all(arrays['cell.'+n].shape == tuple(p.shape) and arrays['cell.'+n].dtype == np.float32
                            and np.isfinite(arrays['cell.'+n]).all() for n, p in model.cell.state_dict().items()), 'finite float32 cell tensors')
            model.cell.load_state_dict({n: torch.from_numpy(arrays['cell.'+n]) for n in model.cell.state_dict()}, strict=True)
            digest = cell_hash(model); check_frozen(model, digest); return model, digest
        anchors = []
        for seed in old.SEEDS:
            candidate, digest = initialize('last_two', seed)
            parent = history.model_for('last_two', seed, linear)
            parent.load_state_dict({n: torch.from_numpy(v) for n, v in cells[seed].items()}, strict=True)
            batch = old.window_batch(fit_data, {k: v[0] for k, v in batches[seed].items()}, 32, 128)
            with torch.no_grad():
                left, right = old.infer(candidate, batch), old.infer(parent, batch)
            old.require(torch.equal(left, right), 'last-two inherited forward parity on first paired FIT batch')
            anchors.append({'seed': seed, 'frozen_cell_sha256': digest, 'exact_parent_forecast': True,
                            'scope': 'first paired FIT batch only; no evaluation data access'})
        old.write_json(output/'parent-parity.json', anchors)
        for identity in identities()[:18]:
            check(); before = time.monotonic(); fit_ns = clock.now_ns(); arm, seed, key = identity['arm'], identity['seed'], identity['key']
            model, digest = initialize(arm, seed)
            def fit_check(ns=fit_ns):
                check(); history.check_deadline(clock, ns, cfg['fit_cap_seconds'], whole=False)
            result = old.train_one(model, fit_data, batches[seed], cfg=cfg, lr=identity['learning_rate'],
                                   folder=output/key, check=fit_check, fit_started=before)
            model.zero_grad(set_to_none=True); model.eval(); check_frozen(model, digest)
            elapsed = (clock.now_ns()-fit_ns)/1e9
            status = result['status'] if elapsed < cfg['fit_cap_seconds'] else 'FAILED'
            if status == 'PASS':
                with np.load(output/key/'optimizer.npz', allow_pickle=False) as a:
                    check_optimizer(model, {k: a[k] for k in a.files}, cfg['updates'])
            record = {**identity, 'fit': result, 'effective_status': status, 'native_fit_seconds': elapsed,
                      'frozen_cell_sha256': digest, 'resources': resource_model(model, arm)}
            fits.append(record); models[key] = model; states[key] = confirmation.parameter_hash(model)
            old.write_json(output/f'completed-fit-{len(fits):02d}.json', record)
            print(json.dumps({'completed_fit': len(fits), 'key': key, 'status': status, 'updates': result['completed_updates']}), flush=True)
        old.require(len(fits) == 18, 'all fresh fits before DEV loads')
        parent_ledger = {r['key']: r for r in read(output/'parent/fits.json')}
        for identity in identities()[18:48]:
            check(); arm, seed, key = identity['arm'], identity['seed'], identity['key']; folder = output/key; folder.mkdir()
            if arm in FIXED:
                model, digest = initialize(arm, seed); np.savez_compressed(folder/'final.npz', **old.weights(model))
                record = {**identity, 'effective_status': 'PASS', 'frozen_cell_sha256': digest, 'resources': resource_model(model, arm)}
            else:
                original = CACHED_ORIGINAL[arm]; source = confirmation.model_key(original, seed)
                arrays = load(plan, 'parent/'+source+'/final.npz'); model = model_for(arm, seed, linear)
                model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()}, strict=True)
                (folder/'final.npz').write_bytes((output/'parent'/source/'final.npz').read_bytes())
                inherited = parent_ledger[source]
                old.require(inherited['effective_status'] == inherited['fit']['status'] == 'PASS', 'successful inherited fit')
                record = {**identity, 'effective_status': 'PASS', 'parent_fit': inherited, 'resources': resource_model(model, arm)}
            model.eval(); fits.append(record); models[key] = model; states[key] = confirmation.parameter_hash(model)
        old.write_json(output/'checkpoint-barrier.json', {'fresh_fit_attempts': 18, 'fixed_models': 9, 'cached_models': 21,
            'exposed_loads_this_run': 0, 'all_evaluation_data_exposed': True, 'state_sha256': states,
            'checkpoints': {k: old.descriptor(output/k/'final.npz') for k in models}})
        old.write_json(output/'fits.json', fits)
        ridges = {}
        for arm, penalty in zip(REFERENCES[:2], (1., 100.), strict=True):
            saved = load(plan, 'parent/'+arm+'.npz'); metadata = read(output/'parent'/(arm+'.json'))
            ridges[arm] = CausalRobotRidge(tuple(saved[f'h{h:03d}'] for h in range(1, 129)), penalty, metadata['fit_rows'])
        def predictor(identity):
            arm, key = identity['arm'], identity['key']
            def predict(batch):
                check()
                return old.infer(models[key], batch).numpy().astype(np.float64) if key in models else (
                    predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges
                    else old.reference_predict(arm, {'linear_frozen': linear}, batch))
            return predict
        timing_batch, attempts = None, []
        fit_lookup = {r['key']: r for r in fits}
        for name in EXPOSED:
            physical = {'name': name, **load(plan, 'data/'+name)}; data = old.normalized_record(physical, norm)
            starts = old.dev_windows(len(data['q']), cfg); old.require(np.array_equal(starts, 64+160*np.arange(22)), 'all22 scheduled windows')
            batch = old.window_batch([data], {'record': np.zeros(22, dtype=np.int64), 'start': starts}, 32, 128)
            np.savez_compressed(output/('dev-windows-'+name+'.npz'), starts=starts, target=batch['target'])
            if timing_batch is None:
                timing_batch = old.window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, 32, 128)
            for identity in identities():
                check(); key = identity['key']; error = None; prediction = None
                if key in fit_lookup and fit_lookup[key]['effective_status'] != 'PASS':
                    error = {'type': 'FailedTrainingAttempt', 'effective_status': fit_lookup[key]['effective_status']}
                else:
                    try:
                        with torch.no_grad(): prediction = predictor(identity)(batch)
                    except old.FitFailure as exc: error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                    except ValueError as exc:
                        if str(exc) != 'nonfinite ridge prediction': raise
                        error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                filename = 'prediction-'+name+'-'+key+'.npz'
                if prediction is not None: np.savez_compressed(output/filename, prediction=prediction)
                common = {'recording': name, 'arm': identity['arm'], 'seed': identity['seed'], 'learning_rate': identity['learning_rate'], 'fit_key': key}
                scored = old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error); rows.extend(scored)
                attempt = {**common, 'status': 'PASS' if all(r['status']=='PASS' for r in scored) else 'FAILED',
                           'errors': [r['error'] for r in scored], 'prediction_file': filename if prediction is not None else None}
                attempts.append(attempt); old.write_json(output/f'completed-prediction-{len(attempts):03d}.json', attempt)
        selection = select(rows, cfg)
        for identity in resource_identities(selection):
            key, arm = identity['key'], identity['arm']; error = None; timing = None
            if identity['origin'] == 'unavailable':
                spec = fit_lookup[f'{arm}-{identity["seed"]}-lr0']['resources']
                resource = {**identity, **spec, 'status': 'UNAVAILABLE', 'error': {'type': 'UnavailableSelectedRecipe'}, 'timing': None}
                resources.append(resource); old.write_json(output/f'completed-timing-{len(resources):02d}.json', resource)
                continue
            if key in models:
                spec = fit_lookup[key]['resources']
            else:
                spec = publication.auditor.storage(arm)
            try:
                timing = history.timed_request(predictor(identity), timing_batch, norm, cfg)
            except old.FitFailure as exc: error = {'type': 'NonfiniteTiming', 'message': str(exc)}
            except ValueError as exc:
                if str(exc) not in (*NUMERIC_ERRORS, 'finite complete timed request', 'nonfinite ridge prediction'): raise
                error = {'type': 'NonfiniteTiming', 'message': str(exc)}
            resource = {**identity, **spec, 'status': 'PASS' if error is None else 'FAILED', 'error': error, 'timing': timing}
            resources.append(resource); old.write_json(output/f'completed-timing-{len(resources):02d}.json', resource)
        after = {k: confirmation.parameter_hash(m) for k, m in models.items()}; old.require(after == states, 'no inference parameter mutation')
        result = evaluate_rule(rows, selection, resources, cfg)
        old.write_json(output/'parameter-checks.json', {'before': states, 'after': after, 'unchanged': True})
        old.write_json(output/'prediction-attempts.json', attempts); old.write_json(output/'resources.json', resources)
        old.write_json(output/'results.json', {'version': VERSION, 'config': cfg, 'selection': selection, 'rows': rows, 'result': result})
        final, final_sha = authenticate(registration); old.require(final == plan and final_sha == sha, 'all original inputs unchanged')
        old.write_json(output/'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}})
        check(); old.write_json(output/'receipt.json', {'status': 'PASS', 'registration_sha256': sha, 'seconds': (clock.now_ns()-start_ns)/1e9,
            'monotonic_seconds': time.monotonic()-started, 'clock': clock.backend, 'fresh_fits': 18, 'fixed_models': 9, 'cached_models': 21,
            'rows': len(rows), 'prediction_attempts': len(attempts), 'timing_attempts': len(resources), 'raw_mat_decodes': 0,
            'saved_fit_loads': 7, 'saved_exposed_loads': 4, 'all_evaluation_data_exposed': True, 'official_test_access': False,
            'scientific_result': result['status']})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        old.write_json(output/'failure.json', {'status':'FAILED', 'type':type(exc).__name__, 'message':str(exc), 'completed_models':len(fits), 'rows':len(rows)})
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--registration', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.registration, args.output)
