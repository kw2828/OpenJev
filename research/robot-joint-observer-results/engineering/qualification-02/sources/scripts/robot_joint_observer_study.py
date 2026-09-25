"""Joint prefix observer/dynamics development, fixed recipes and inherited evidence."""
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
import publish_robot_position_observer as publication
import robot_position_observer_study as parent
import torch

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.robot_joint_observer import JointObserver
from openjev.research.robot_position_observer import gradient_diagnostics
from openjev.research.suspend_clock import SuspendClock

old, history, confirmation = parent.old, parent.history, parent.confirmation
ROOT = old.ROOT
VERSION = 'robot-joint-observer-study-v1'
ARMS = ('joint_observer_long', 'joint_observer_short', 'continued_last_two', 'continued_temporal')
MODES = dict(zip(ARMS, ('long', 'short', 'last_two', 'temporal'), strict=True))
CANDIDATE, ALIAS = ARMS[0], 'frozen_position_lr001'
REFERENCES, DEV, EXPOSED = parent.REFERENCES, parent.DEV, parent.EXPOSED
INHERITED_RATES = {parent.CANDIDATE: .003, parent.POSITION_FIXED: None, **parent.INHERITED_RATES}
ELIGIBLE = (*ARMS, *parent.ELIGIBLE, ALIAS)
FAMILIES = (*ARMS, *parent.FAMILIES, ALIAS)
SIMPLE = ARMS[1:]
SOURCES = (*parent.SOURCES, 'scripts/publish_robot_position_observer.py',
           'tests/test_publish_robot_position_observer.py', 'src/openjev/research/robot_joint_observer.py',
           'tests/test_robot_joint_observer.py', 'scripts/robot_joint_observer_study.py',
           'tests/test_robot_joint_observer_study.py', 'research/robot-joint-observer-protocol.md',
           'scripts/audit_robot_joint_observer.py', 'tests/test_audit_robot_joint_observer.py')
LAUNCHER = 'scripts/launch_robot_joint_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:], LAUNCHER)
CONDITIONS, NUMERIC_ERRORS = parent.CONDITIONS, parent.NUMERIC_ERRORS
pin, read = parent.pin, parent.read
native_norm_record, aggregate_diagnostics = parent.native_norm_record, parent.aggregate_diagnostics
COMMON = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
          'causal_ridge_100.npz', 'causal_ridge_100.json', *(f'batches-{s}.npz' for s in old.SEEDS))
METADATA = ('results.json', 'fits.json', 'resources.json', 'prediction-attempts.json')


def config():
    cfg = parent.config()
    for key in ('learned', 'fixed', 'diagnostic_probes', 'new_fixed_models'):
        cfg.pop(key, None)
    cfg.update(version=VERSION, arms=list(ARMS), learning_rates=[.001], families=list(FAMILIES),
               eligible_families=list(ELIGIBLE), inherited_rates=dict(INHERITED_RATES),
               fixed_rates=dict.fromkeys(ARMS, .001), alias=ALIAS, alias_source_arm=parent.CANDIDATE,
               alias_learning_rate=.001, new_fits=12, inherited_models=45, final_models=57,
               inherited_rows=488, inherited_prediction_attempts=244, input_count=69,
               wall_cap_seconds=7200.,
               selection_scope='No new rate selection; four fresh rates fixed .001 and inherited recipes frozen',
               gradient_clipping='unchanged native float32 norm; float64 diagnostic only',
               optimizer_initialization='fresh empty Adam for every fresh fit; no inherited optimizer')
    return cfg


def identities():
    return [{'key': f'{a}-{s}-lr0', 'arm': a, 'seed': s, 'learning_rate': .001, 'origin': 'fresh'}
            for a in ARMS for s in old.SEEDS]


def inherited_identities():
    return [*parent.identities(), *[r for r in parent.inherited_identities() if r['arm'] not in REFERENCES]]


def alias_identities():
    return [{'key': f'{ALIAS}-{s}-lr0', 'arm': ALIAS, 'seed': s, 'learning_rate': .001,
             'origin': 'inherited_alias', 'model_key': f'{parent.CANDIDATE}-{s}-lr0'} for s in old.SEEDS]


def expected_input_names():
    return ({'parent/'+n for n in (*COMMON, *METADATA)}
            | {'parent/'+r['key']+'/final.npz' for r in inherited_identities()}
            | {'data/'+n for n in (*config()['partitions']['fit'], *EXPOSED)})


def expected_inputs(prior):
    old.require(prior['audit']['results']['result']['status'] == 'POSITION_OBSERVER_DEVELOPMENT_FAIL', 'preserve failed parent')
    old.require(prior['audit']['results']['selection']['selected_rates'] == {parent.CANDIDATE: .003}, 'frozen parent selected rate')
    result = {}
    for name in expected_input_names():
        if name.startswith('data/') or name in {'parent/'+n for n in COMMON}:
            result[name] = prior['plan']['inputs'][name]
        else:
            relative = name.removeprefix('parent/')
            item = pin(ROOT/publication.STUDY/relative)
            old.require({k: item[k] for k in ('sha256', 'bytes')} == prior['inventory'][relative], 'audited immediate parent payload')
            result[name] = item
    old.require(len(result) == 69, 'exact69 inputs')
    return result


def authenticate(registration):
    raw = Path(registration).read_bytes(); plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact joint-observer config')
    old.require(set(plan['sources']) == set(SOURCES) and len(SOURCES) == 63, 'all63 scientific sources')
    for name, item in plan['sources'].items():
        old.require(old.descriptor(ROOT/name) == item, 'source drift: '+name)
    old.require(plan['launcher'] == old.descriptor(ROOT/LAUNCHER), 'launcher drift')
    old.require(all(os.environ.get(k) == '1' for k in old.THREADS), 'single-thread environment')
    confirmation._pin(plan['qualification']); q = read(plan['qualification']['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_joint_observer.py',
                 'tests/test_robot_joint_observer_study.py', 'tests/test_audit_robot_joint_observer.py']]
    old.require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
                and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(old.THREADS, '1')
                and [r['command'] for r in q['commands']] == commands, 'exact prefit qualification')
    for row in q['commands']:
        old.require(row['returncode'] == 0 and old.descriptor(row['log'])['sha256'] == row['sha256'], 'qualification original log')
    prior = publication.authenticate(ROOT/publication.STUDY, ROOT/publication.AUDIT, ROOT/publication.ENGINEERING)
    old.require(plan['inputs'] == expected_inputs(prior), 'exact69 admitted parent inputs; no TEST')
    for item in plan['inputs'].values(): confirmation._pin(item)
    old.require(plan['parent_publication_manifest'] == pin(ROOT/publication.OUTPUT/'manifest.json')
                and plan['parent_audit'] == pin(ROOT/publication.AUDIT), 'closed parent audit/publication')
    return plan, hashlib.sha256(raw).hexdigest()


def load(plan, name):
    old.require(name in expected_input_names() and name in plan['inputs'], 'registered numerical input only')
    item = plan['inputs'][name]; confirmation._pin(item)
    old.require(item['path'].endswith('.npz'), 'only NPZ numerical decoder')
    with np.load(item['path'], allow_pickle=False) as bank:
        arrays = {k: bank[k].copy(order='K') for k in bank.files}
    if name.startswith('data/'):
        old.require(set(arrays) == {'q', 'u', 'raw_indices'} and all(arrays[k].dtype == np.float64
                    and arrays[k].shape == (3636, 6) and np.isfinite(arrays[k]).all() for k in ('q', 'u'))
                    and arrays['raw_indices'].dtype == np.int64
                    and np.array_equal(arrays['raw_indices'], np.arange(0, 90881, 25)), 'complete processed recording schema')
    return arrays


class JointAdapter(JointObserver):
    def condition(self, q, u, diagnostics=None):
        return parent.parent.ObserverAdapter.numeric(super().condition, q, u, diagnostics)

    def forward(self, u, state):
        return parent.parent.ObserverAdapter.numeric(super().forward, u, state)


def model_for(arm, seed, linear=None):
    return JointAdapter(seed, MODES[arm]) if arm in ARMS else parent.model_for(arm, seed, linear)


def resource_model(model, arm):
    return parent.resource_model(model, arm)


def check_optimizer(model, arrays, completed_updates):
    names = dict(model.named_parameters())
    old.require(all(p.requires_grad for p in names.values()), 'all joint-model parameters trainable')
    old.require(set(arrays) == {n+'/'+s for n in names for s in ('step', 'exp_avg', 'exp_avg_sq')}, 'all-parameter Adam state')
    for n, p in names.items():
        step = arrays[n+'/step']
        old.require(step.shape == () and np.isfinite(step).all() and float(step) == completed_updates, 'exact Adam update count')
        for suffix in ('exp_avg', 'exp_avg_sq'):
            a = arrays[n+'/'+suffix]
            old.require(a.shape == tuple(p.shape) and a.dtype == np.float32 and np.isfinite(a).all(), 'finite Adam moments')


def row_arm(row):
    return ALIAS if row['arm'] == parent.CANDIDATE and row['learning_rate'] == .001 else row['arm']


def selected(row, selection):
    arm = row['arm']
    if arm in ARMS: return row['learning_rate'] == .001
    if arm in REFERENCES: return True
    if arm == parent.CANDIDATE and row['learning_rate'] == .001: return True
    return arm in INHERITED_RATES and row['learning_rate'] == INHERITED_RATES[arm]


def resource_identities(selection=None):
    prior = parent.resource_identities({'selected_rates': {parent.CANDIDATE: .003}})
    return [*identities(), *prior, *alias_identities()]


def validate_rows(rows, cfg):
    roster = {r['key']: r for r in (*parent.parent.identities(), *parent.identities(), *identities())}
    expected = {(n, k, h) for n in EXPOSED for k in roster for h in (64, 128)}
    ids = [(r['recording'], r['fit_key'], r['horizon']) for r in rows]
    old.require(len(ids) == len(set(ids)) == 584 and set(ids) == expected, 'all584 scheduled score rows; no duplicated alias')
    for r in rows:
        old.require(all(r[k] == roster[r['fit_key']][k] for k in ('arm', 'seed', 'learning_rate')), 'score identity join')
        old.require(r['status'] in ('PASS', 'FAILED'), 'declared score status')
        if r['status'] == 'PASS':
            old.require(r['error'] is None and old.valid_metric_row(r) and r['metrics']['windows'] == 22
                        and r['metrics']['scalars'] == 22*r['horizon']*6 and r['metrics']['horizon'] == r['horizon'], 'complete metric geometry')
        else:
            old.require(r['error'] is not None and r['metrics'] is None, 'explicit failed metric')
    old.require(cfg == config(), 'immutable joint-observer config')


def select(rows, cfg):
    validate_rows(rows, cfg)
    return {'fixed_rates': dict.fromkeys(ARMS, .001), 'inherited_rates': dict(INHERITED_RATES),
            'alias': {'arm': ALIAS, 'source_arm': parent.CANDIDATE, 'learning_rate': .001}, 'scope': cfg['selection_scope']}


def evaluate_rule(rows, selection, resources, cfg):
    validate_rows(rows, cfg)
    old.require(selection == select(rows, cfg), 'fixed recipes; no selection')
    details, means, costs = {}, {}, {}
    for name in EXPOSED:
        values = {}
        for arm in ELIGIBLE:
            subset = [r for r in rows if r['recording'] == name and row_arm(r) == arm and r['horizon'] == 128 and selected(r, selection)]
            if len(subset) == (1 if arm in REFERENCES else 3) and all(old.valid_metric_row(r) for r in subset):
                values[arm] = float(np.mean([r['metrics']['standardized_rmse'] for r in subset]))
        details[name] = {'means': values}
    fields = ('key', 'arm', 'seed', 'learning_rate')
    expected = {tuple(r[k] for k in fields) for r in resource_identities()}
    actual = [tuple(r[k] for k in fields) for r in resources]
    old.require(len(actual) == len(set(actual)) == 61 and set(actual) == expected, 'all61 cost identities')
    for arm in ELIGIBLE:
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
            if good: times.append(duration); sizes.append(sum(parts))
        costs[arm] = {'latency': float(np.median(times)), 'bytes': max(sizes)} if good else None
    frontier = all(a in means and costs[a] is not None for a in ELIGIBLE)
    complete = frontier and all(old.valid_metric_row(r) for r in rows if selected(r, selection))
    candidate = means.get(CANDIDATE); controls = [means.get(a) for a in ELIGIBLE if a != CANDIDATE]
    best = min(controls) if all(v is not None for v in controls) else None
    gain = candidate is not None and best is not None and best > 0 and candidate <= .95*best
    no_harm = all(all(a in d['means'] for a in (*SIMPLE, CANDIDATE)) and d['means'][CANDIDATE]
                  <= 1.02*min(d['means'][a] for a in SIMPLE) for d in details.values())
    left, base = costs[CANDIDATE], costs['continued_last_two']
    latency = left is not None and base is not None and left['latency'] <= 1.5*base['latency']
    dominators = []
    if frontier:
        point = (candidate, left['latency'], left['bytes'])
        for arm in ELIGIBLE:
            if arm == CANDIDATE: continue
            other = (means[arm], costs[arm]['latency'], costs[arm]['bytes'])
            if all(x <= y for x,y in zip(other, point, strict=True)) and any(x < y for x,y in zip(other, point, strict=True)):
                dominators.append(arm)
    conditions = [{'name': n, 'passed': bool(v)} for n,v in zip(CONDITIONS, (complete, gain, no_harm, latency, frontier and not dominators), strict=True)]
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'JOINT_OBSERVER_DEVELOPMENT_PASS' if passed == 5 else 'JOINT_OBSERVER_DEVELOPMENT_FAIL',
            'passed': passed, 'total': 5, 'conditions': conditions, 'details': details, 'equal_file_means': means,
            'costs': costs, 'best_control_mean': best, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'All four files exposed development; no confirmation, novelty or control claim'}


def infer_diagnostic(model, batch, prefix):
    q = torch.from_numpy(np.asarray(batch['q_context'], dtype=np.float32))
    u = torch.from_numpy(np.asarray(batch['u_context'], dtype=np.float32))
    future = torch.from_numpy(np.asarray(batch['future_u'], dtype=np.float32))
    return model(future, model.condition(q, u, diagnostics=prefix))[0]


def gradient_snapshot(model):
    names = dict(model.named_parameters())
    old.require(all(p.requires_grad and p.grad is not None for p in names.values()),
                'every trainable parameter has a present gradient')
    old.require(all(p.grad.device.type == 'cpu' and p.grad.dtype == torch.float32
                    and tuple(p.grad.shape) == tuple(p.shape) for p in names.values()), 'raw float32 gradient schema')
    return {n: p.grad.detach().cpu().numpy().copy() for n,p in names.items()}


def train_one(model, records, batches, *, cfg, lr, folder, check=lambda: None, fit_started=None):
    """Original native optimizer path; all parameters trainable with observational diagnostics."""
    started = time.monotonic() if fit_started is None else fit_started
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'initial.npz', **old.weights(model))
    old.require(all(p.requires_grad for p in model.parameters()), 'all joint parameters trainable')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    trace, diagnostics, last_gradient = [], [], {}
    error, status, attempt = None, 'PASS', None
    loop_started = time.monotonic()
    try:
        for update in range(cfg['updates']):
            attempt = {'update': update+1, 'prefix': {}, 'gradient': None,
                       'native_norm': None, 'status': 'PASS', 'error': None}
            diagnostics.append(attempt)
            last_gradient = {}
            check()
            if time.monotonic()-started > cfg['fit_cap_seconds']:
                raise old.FitFailure('single-fit wall cap exceeded')
            choices = {key: value[update] for key, value in batches.items()}
            batch = old.window_batch(records, choices, cfg['context'], cfg['train_horizon'])
            optimizer.zero_grad(set_to_none=True)
            prediction = infer_diagnostic(model, batch, attempt['prefix'])
            target = torch.from_numpy(np.asarray(batch['target'], dtype=np.float32))
            loss = (prediction-target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise old.FitFailure('nonfinite autoregressive training loss')
            loss.backward()
            last_gradient = gradient_snapshot(model)
            attempt['gradient'] = gradient_diagnostics(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=False)
            last_gradient['native_norm'] = norm.detach().cpu().numpy().copy()
            attempt['native_norm'] = native_norm_record(norm)
            if not bool(torch.isfinite(norm)):
                raise old.FitFailure('nonfinite training gradient norm')
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise old.FitFailure('nonfinite updated parameters')
            if not all(np.isfinite(value).all() for value in old.optimizer_arrays(model, optimizer).values()):
                raise old.FitFailure('nonfinite updated Adam state')
            trace.append({'update': update+1, 'loss': float(loss.detach()),
                          'gradient_norm_before_clip': float(norm.detach())})
            check()
            if time.monotonic()-started > cfg['fit_cap_seconds']:
                raise old.FitFailure('single-fit wall cap exceeded')
    except old.FitFailure as exc:
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
    except ValueError as exc:
        # Preserve the original trainer's one exact legacy numerical guard.
        # JointAdapter already translates its own qualified numeric guards.
        if str(exc) != 'nonfinite joint-coupling output; no clipping or repair':
            status, error = 'FATAL', {'type': type(exc).__name__, 'message': str(exc)}
            raise
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
    except BaseException as exc:
        status, error = 'FATAL', {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        optimizer_seconds = time.monotonic()-loop_started
        if attempt is not None and status != 'PASS':
            attempt.update(status=status, error=error)
        np.savez_compressed(folder/'final.npz', **old.weights(model))
        np.savez_compressed(folder/'optimizer.npz', **old.optimizer_arrays(model, optimizer))
        old.write_json(folder/'diagnostics.json', diagnostics)
        np.savez_compressed(folder/'last-gradient.npz', **last_gradient)
        old.write_json(folder/'diagnostic-summary.json', aggregate_diagnostics(diagnostics))
        old.write_json(folder/'trace.json', trace)
        fit_seconds = time.monotonic()-started
        if status == 'PASS' and fit_seconds > cfg['fit_cap_seconds']:
            status, error = 'FAILED', {'type': 'FitFailure', 'message': 'single-fit wall cap exceeded during preservation'}
        receipt = {'status': status, 'error': error, 'completed_updates': len(trace),
                   'requested_updates': cfg['updates'], 'learning_rate': lr,
                   'optimizer_seconds': optimizer_seconds, 'fit_seconds': fit_seconds,
                   'fit_cap_scope': 'construction,optimizer setup,loop and checkpoint preservation through trace;receipt serialization follows',
                   'timing_scope': 'optimizer loop including batch construction and finite checks, excluding model construction and saved files',
                   'diagnostics_scope': 'Per-attempt detached prefix maxima and PRE-clipping gradient summaries included in loop time; '
                                        'only the last attempt retains raw gradient/native norm arrays; no historical backward replay',
                   'files': {p.name: old.descriptor(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        old.write_json(folder/'fit-receipt.json', receipt)
    return receipt


def initialize(model, arrays):
    old.require(set(arrays) == {'cell.'+n for n in model.cell.state_dict()}, 'exact common last-two cell schema')
    old.require(all(arrays['cell.'+n].shape == tuple(p.shape) and arrays['cell.'+n].dtype == np.float32
                    and np.isfinite(arrays['cell.'+n]).all() for n,p in model.cell.state_dict().items()), 'finite float32 common cell')
    model.cell.load_state_dict({n: torch.from_numpy(arrays['cell.'+n]) for n in model.cell.state_dict()}, strict=True)
    old.require(all(p.requires_grad and p.grad is None for p in model.parameters()), 'fresh joint trainable weights without gradients')
    if hasattr(model, 'gain'):
        expected_gain = torch.cat((torch.eye(6, dtype=torch.float32), torch.zeros((6, 6), dtype=torch.float32)))
        old.require(torch.equal(model.gain, expected_gain), 'exact initial [I;0] gain')
    if hasattr(model, 'head'):
        old.require(all(torch.count_nonzero(p) == 0 for p in model.head.parameters()), 'exact zero temporal head')
    return {'cell_sha256': parent.cell_hash(model), 'parameter_sha256': confirmation.parameter_hash(model),
            'optimizer_state_entries': 0, 'scope': 'Common parent cell weights only; fresh empty Adam; no inherited optimizer'}


def timing_attempt(identity, fit_lookup, predictor, timing_batch, norm, cfg):
    key, arm = identity.get('model_key', identity['key']), identity['arm']
    spec = fit_lookup[key]['resources'] if key in fit_lookup else publication.auditor.storage(arm)
    if arm in ARMS and fit_lookup[key]['effective_status'] != 'PASS':
        return {**identity, **spec, 'status': 'UNAVAILABLE',
                'error': {'type': 'FailedTrainingAttempt', 'effective_status': fit_lookup[key]['effective_status']}, 'timing': None}
    error, timing = None, None
    try:
        timing = history.timed_request(predictor(identity), timing_batch, norm, cfg)
    except old.FitFailure as exc: error = {'type': 'NonfiniteTiming', 'message': str(exc)}
    except ValueError as exc:
        if str(exc) not in (*NUMERIC_ERRORS, 'finite complete timed request', 'nonfinite ridge prediction'): raise
        error = {'type': 'NonfiniteTiming', 'message': str(exc)}
    return {**identity, **spec, 'status': 'PASS' if error is None else 'FAILED', 'error': error, 'timing': timing}


def run(registration, output):
    clock = SuspendClock(); start_ns, started = clock.now_ns(), time.monotonic()
    plan, sha = authenticate(registration); cfg = config(); output = Path(output)
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    fits, models, resources, states, initialization = [], {}, [], {}, {}
    rows, attempts = [], []
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
        for name in sorted(plan['inputs']):
            if not name.startswith('data/'): copy_input(name)
        rows = read(output/'parent/results.json')['rows'].copy()
        attempts = read(output/'parent/prediction-attempts.json')
        old.require(len(rows) == 488 and len(attempts) == 244, 'complete unchanged parent outcomes')
        norm = load(plan, 'parent/normalizers.npz'); linear = load(plan, 'parent/linear.npz')['coefficient']
        physical_fit = [{'name': n, **load(plan, 'data/'+n)} for n in cfg['partitions']['fit']]
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        batches = {s: load(plan, f'parent/batches-{s}.npz') for s in old.SEEDS}
        for seed, bank in batches.items():
            expected = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(bank) == set(expected) and all(np.array_equal(bank[k], expected[k]) for k in bank), 'same paired4096 batches')
        cells = {s: load(plan, f'parent/last_two-{s}-fixed/final.npz') for s in old.SEEDS}
        for identity in identities():
            check(); before = time.monotonic(); fit_ns = clock.now_ns()
            arm, seed, key = identity['arm'], identity['seed'], identity['key']
            model = model_for(arm, seed, linear)
            initialization[key] = initialize(model, cells[seed])
            def fit_check(ns=fit_ns):
                check(); history.check_deadline(clock, ns, cfg['fit_cap_seconds'], whole=False)
            result = train_one(model, fit_data, batches[seed], cfg=cfg, lr=.001, folder=output/key,
                               check=fit_check, fit_started=before)
            model.zero_grad(set_to_none=True); model.eval()
            elapsed = (clock.now_ns()-fit_ns)/1e9
            status = result['status'] if elapsed < cfg['fit_cap_seconds'] else 'FAILED'
            native_error = None if elapsed < cfg['fit_cap_seconds'] else {'type': 'FitFailure', 'message': 'native fit cap includes preservation'}
            if status == 'PASS':
                with np.load(output/key/'optimizer.npz', allow_pickle=False) as bank:
                    check_optimizer(model, {k: bank[k] for k in bank.files}, cfg['updates'])
            record = {**identity, 'fit': result, 'effective_status': status, 'native_fit_seconds': elapsed,
                      'native_fit_error': native_error, 'initialization': initialization[key], 'resources': resource_model(model, arm)}
            fits.append(record); models[key] = model; states[key] = confirmation.parameter_hash(model)
            old.write_json(output/f'completed-fit-{len(fits):02d}.json', record)
            print(json.dumps({'completed_fit': len(fits), 'key': key, 'status': status, 'updates': result['completed_updates']}), flush=True)
        old.require(len(fits) == 12, 'all twelve new attempts before evaluation loads')
        for seed in old.SEEDS:
            entries = [initialization[f'{a}-{seed}-lr0'] for a in ARMS]
            old.require(len({r['cell_sha256'] for r in entries}) == 1, 'same initial parent cell across all four arms')
        old.write_json(output/'initialization-checks.json', initialization)
        parent_fits = {r['key']: r for r in read(output/'parent/fits.json')}
        old.require(set(parent_fits) == {r['key'] for r in inherited_identities()}, 'all45 parent model records')
        for identity in inherited_identities():
            check(); arm, seed, key = identity['arm'], identity['seed'], identity['key']
            arrays = load(plan, 'parent/'+key+'/final.npz'); model = model_for(arm, seed, linear)
            model.load_state_dict({n: torch.from_numpy(v) for n,v in arrays.items()}, strict=True)
            inherited = parent_fits[key]
            old.require(inherited['effective_status'] == 'PASS', 'successful inherited checkpoint')
            folder = output/key; folder.mkdir(); (folder/'final.npz').write_bytes((output/'parent'/key/'final.npz').read_bytes())
            fits.append(inherited); model.eval(); models[key] = model; states[key] = confirmation.parameter_hash(model)
        old.require(len(models) == len(fits) == 57, 'all57 checkpoints before exposed arrays')
        old.write_json(output/'checkpoint-barrier.json', {'fresh_fit_attempts': 12, 'inherited_models': 45,
            'exposed_loads_this_run': 0, 'all_evaluation_data_exposed': True, 'state_sha256': states,
            'checkpoints': {k: old.descriptor(output/k/'final.npz') for k in models}})
        old.write_json(output/'fits.json', fits)
        ridges = {}
        for arm, penalty in zip(REFERENCES[:2], (1., 100.), strict=True):
            saved = load(plan, 'parent/'+arm+'.npz'); metadata = read(output/'parent'/(arm+'.json'))
            ridges[arm] = CausalRobotRidge(tuple(saved[f'h{h:03d}'] for h in range(1, 129)), penalty, metadata['fit_rows'])
        def predictor(identity):
            arm, key = identity['arm'], identity.get('model_key', identity['key'])
            def predict(batch):
                check()
                return old.infer(models[key], batch).numpy().astype(np.float64) if key in models else (
                    predict_ridge(ridges[arm], batch['q_context'], batch['u_context'], batch['future_u']) if arm in ridges
                    else old.reference_predict(arm, {'linear_frozen': linear}, batch))
            return predict
        timing_batch = None
        fit_lookup = {r['key']: r for r in fits}
        for name in EXPOSED:
            physical = {'name': name, **load(plan, 'data/'+name)}; data = old.normalized_record(physical, norm)
            starts = old.dev_windows(len(data['q']), cfg); old.require(np.array_equal(starts, 64+160*np.arange(22)), 'all22 scheduled windows')
            batch = old.window_batch([data], {'record': np.zeros(22, dtype=np.int64), 'start': starts}, 32, 128)
            np.savez_compressed(output/('dev-windows-'+name+'.npz'), starts=starts, target=batch['target'])
            if timing_batch is None:
                timing_batch = old.window_batch([physical], {'record': np.zeros(1, dtype=np.int64), 'start': starts[:1]}, 32, 128)
            for identity in identities():
                check(); key = identity['key']; error, prediction = None, None
                if fit_lookup[key]['effective_status'] != 'PASS':
                    error = {'type': 'FailedTrainingAttempt', 'effective_status': fit_lookup[key]['effective_status']}
                else:
                    try:
                        with torch.no_grad(): prediction = predictor(identity)(batch)
                    except old.FitFailure as exc: error = {'type': 'NonfiniteEvaluation', 'message': str(exc)}
                filename = 'prediction-'+name+'-'+key+'.npz'
                if prediction is not None: np.savez_compressed(output/filename, prediction=prediction)
                common = {'recording': name, 'arm': identity['arm'], 'seed': identity['seed'], 'learning_rate': identity['learning_rate'], 'fit_key': key}
                scored = old.scored_rows(common, prediction, batch['target'], norm['q_std'], cfg, error); rows.extend(scored)
                attempt = {**common, 'status': 'PASS' if all(r['status']=='PASS' for r in scored) else 'FAILED',
                           'errors': [r['error'] for r in scored], 'prediction_file': filename if prediction is not None else None}
                attempts.append(attempt); old.write_json(output/f'completed-prediction-{len(attempts)-244:03d}.json', attempt)
        selection = select(rows, cfg)
        for identity in resource_identities():
            resource = timing_attempt(identity, fit_lookup, predictor, timing_batch, norm, cfg)
            resources.append(resource); old.write_json(output/f'completed-timing-{len(resources):02d}.json', resource)
        after = {k: confirmation.parameter_hash(m) for k,m in models.items()}
        old.require(after == states, 'no inference parameter mutation')
        result = evaluate_rule(rows, selection, resources, cfg)
        old.write_json(output/'parameter-checks.json', {'before': states, 'after': after, 'unchanged': True})
        old.write_json(output/'prediction-attempts.json', attempts); old.write_json(output/'resources.json', resources)
        old.write_json(output/'results.json', {'version': VERSION, 'config': cfg, 'selection': selection, 'rows': rows, 'result': result})
        final, final_sha = authenticate(registration); old.require(final == plan and final_sha == sha, 'all original evidence unchanged')
        old.write_json(output/'manifest.json', {'files': {str(p.relative_to(output)): old.descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}})
        check(); old.write_json(output/'receipt.json', {'status': 'PASS', 'registration_sha256': sha, 'seconds': (clock.now_ns()-start_ns)/1e9,
            'monotonic_seconds': time.monotonic()-started, 'clock': clock.backend, 'fresh_fits': 12, 'inherited_models': 45,
            'rows': len(rows), 'prediction_attempts': len(attempts), 'timing_attempts': len(resources), 'raw_mat_decodes': 0,
            'saved_fit_loads': 7, 'saved_exposed_loads': 4, 'inherited_forecast_regenerations': 0,
            'all_evaluation_data_exposed': True, 'official_test_access': False, 'scientific_result': result['status']})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        old.write_json(output/'failure.json', {'status': 'FAILED', 'type': type(exc).__name__, 'message': str(exc), 'completed_models': len(fits), 'rows': len(rows)})
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--registration', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.registration, args.output)
