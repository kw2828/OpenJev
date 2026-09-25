"""Position-only initial observer gain, unchanged optimizer, exposed development."""
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
import publish_robot_observer_study as publication
import robot_observer_study as parent
import torch

from openjev.research.causal_robot_ridge import CausalRobotRidge
from openjev.research.causal_robot_ridge import predict as predict_ridge
from openjev.research.robot_position_observer import FrozenPositionObserver, gradient_diagnostics
from openjev.research.suspend_clock import SuspendClock

old, history, confirmation = parent.old, parent.history, parent.confirmation
ROOT = old.ROOT
VERSION = 'robot-position-observer-study-v1'
CANDIDATE, POSITION_FIXED = 'observer_position', 'observer_position_fixed'
LEARNED, FIXED = (CANDIDATE,), (POSITION_FIXED,)
REFERENCES, DEV, EXPOSED = parent.REFERENCES, parent.DEV, parent.EXPOSED
PARENT_ARMS = tuple(a for a in (*parent.LEARNED, *parent.FIXED, *parent.CACHED) if a != 'observer_learned')
INHERITED_RATES = {'local_affine': .003, 'temporal_affine': .001,
                   **dict.fromkeys(parent.FIXED), **parent.CACHED_RATES}
ELIGIBLE = (CANDIDATE, POSITION_FIXED, *PARENT_ARMS, *REFERENCES)
FAMILIES = (CANDIDATE, POSITION_FIXED, *parent.FAMILIES)
SIMPLE = (*parent.SIMPLE, POSITION_FIXED)
SOURCES = (*parent.SOURCES, 'scripts/publish_robot_observer_study.py',
           'tests/test_publish_robot_observer_study.py', 'src/openjev/research/robot_position_observer.py',
           'tests/test_robot_position_observer.py', 'scripts/robot_position_observer_study.py',
           'tests/test_robot_position_observer_study.py', 'research/robot-position-observer-protocol.md',
           'scripts/audit_robot_position_observer.py', 'tests/test_audit_robot_position_observer.py')
LAUNCHER = 'scripts/launch_robot_position_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:], LAUNCHER)
CONDITIONS, NUMERIC_ERRORS = parent.CONDITIONS, parent.NUMERIC_ERRORS
pin, read = parent.pin, parent.read
cell_hash, check_frozen, check_optimizer = parent.cell_hash, parent.check_frozen, parent.check_optimizer


def config():
    cfg = parent.config()
    for key in ('cached', 'cached_rates'):
        del cfg[key]
    cfg.update(version=VERSION, learned=list(LEARNED), fixed=list(FIXED), families=list(FAMILIES),
               eligible_families=list(ELIGIBLE), inherited_rates=dict(INHERITED_RATES),
               wall_cap_seconds=7200., diagnostic_probes=7, inherited_rows=416,
               inherited_prediction_attempts=208, new_fits=6, new_fixed_models=3,
               input_count=61, gradient_clipping='unchanged native float32 norm; float64 diagnostic only')
    return cfg


def identities():
    return ([{'key': f'{CANDIDATE}-{s}-lr{i}', 'arm': CANDIDATE, 'seed': s,
              'learning_rate': rate, 'origin': 'fresh'} for s in old.SEEDS for i, rate in enumerate((.001, .003))]
            + [{'key': f'{POSITION_FIXED}-{s}-fixed', 'arm': POSITION_FIXED, 'seed': s,
                'learning_rate': None, 'origin': 'fixed'} for s in old.SEEDS])


def inherited_identities():
    return [r for r in parent.identities() if r['arm'] in REFERENCES or
            (r['arm'] in PARENT_ARMS and r['learning_rate'] == INHERITED_RATES[r['arm']])]


def expected_input_names():
    common = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
              'causal_ridge_100.npz', 'causal_ridge_100.json', *(f'batches-{s}.npz' for s in old.SEEDS))
    metadata = ('results.json', 'fits.json', 'resources.json', 'prediction-attempts.json')
    return ({'parent/'+n for n in (*common, *metadata)}
            | {'parent/'+r['key']+'/final.npz' for r in inherited_identities() if r['arm'] not in REFERENCES}
            | {'data/'+n for n in (*config()['partitions']['fit'], *EXPOSED)}
            | {'diagnostic/observer_learned-8103-lr0/final.npz'})


def expected_inputs(prior):
    """Resolve and pin original audited bytes without decoding measurements."""
    old.require(prior['audit']['results']['result']['status'] == 'OBSERVER_DEVELOPMENT_FAIL', 'preserve failed parent')
    selection = prior['audit']['results']['selection']['selected_rates']
    old.require(selection == {'local_affine': .003, 'temporal_affine': .001, 'observer_learned': None},
                'exact frozen parent selection')
    result = {}
    folder = ROOT/publication.STUDY
    for name in expected_input_names():
        common = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
                  'causal_ridge_100.npz', 'causal_ridge_100.json', *(f'batches-{s}.npz' for s in old.SEEDS))
        if name.startswith('data/') or name in {'parent/'+n for n in common}:
            result[name] = prior['plan']['inputs'][name]
        else:
            relative = name.removeprefix('parent/').removeprefix('diagnostic/')
            item = pin(folder/relative)
            old.require({k: item[k] for k in ('sha256', 'bytes')} == prior['inventory'][relative], 'audited parent payload')
            result[name] = item
    old.require(len(result) == 61, 'exact61 inherited inputs')
    return result


def authenticate(registration):
    raw = Path(registration).read_bytes(); plan = json.loads(raw)
    old.require(plan['version'] == VERSION and plan['config'] == config(), 'exact position-observer config')
    old.require(set(plan['sources']) == set(SOURCES) and len(SOURCES) == 54, 'all54 inherited/new sources')
    for name, item in plan['sources'].items():
        old.require(old.descriptor(ROOT/name) == item, 'source drift: '+name)
    old.require(plan['launcher'] == old.descriptor(ROOT/LAUNCHER), 'launcher drift')
    old.require(all(os.environ.get(k) == '1' for k in old.THREADS), 'single-thread environment')
    confirmation._pin(plan['qualification']); q = read(plan['qualification']['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_position_observer.py',
                 'tests/test_robot_position_observer_study.py', 'tests/test_audit_robot_position_observer.py']]
    old.require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
                and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(old.THREADS, '1')
                and [r['command'] for r in q['commands']] == commands, 'exact prefit qualification')
    for row in q['commands']:
        old.require(row['returncode'] == 0 and old.descriptor(row['log'])['sha256'] == row['sha256'], 'qualification original log')
    prior = publication.authenticate(ROOT/publication.STUDY, ROOT/publication.AUDIT, ROOT/publication.ENGINEERING)
    old.require(plan['inputs'] == expected_inputs(prior), 'exact61 audited inputs; no TEST')
    for item in plan['inputs'].values(): confirmation._pin(item)
    old.require(plan['parent_publication_manifest'] == pin(ROOT/publication.OUTPUT/'manifest.json')
                and plan['parent_audit'] == pin(ROOT/publication.AUDIT), 'closed agreeing parent audit/publication')
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


class PositionAdapter(FrozenPositionObserver):
    def condition(self, q, u, diagnostics=None):
        return parent.ObserverAdapter.numeric(super().condition, q, u, diagnostics)

    def forward(self, u, state):
        return parent.ObserverAdapter.numeric(super().forward, u, state)


def model_for(arm, seed, linear=None):
    if arm in (CANDIDATE, POSITION_FIXED):
        return PositionAdapter(seed, 'learned' if arm == CANDIDATE else 'fixed')
    old.require(arm in PARENT_ARMS, 'eligible inherited family')
    return parent.model_for(arm, seed, linear)


def resource_model(model, arm):
    return parent.resource_model(model, arm)


def selected(row, selection):
    arm = row['arm']
    if arm == CANDIDATE:
        return selection['selected_rates'][CANDIDATE] is not None and row['learning_rate'] == selection['selected_rates'][CANDIDATE]
    if arm == POSITION_FIXED or arm in REFERENCES:
        return True
    return arm in PARENT_ARMS and row['learning_rate'] == INHERITED_RATES[arm]


def resource_identities(selection):
    rows = [r for r in identities() if selected(r, selection)]
    if selection['selected_rates'][CANDIDATE] is None:
        rows += [{'key': f'{CANDIDATE}-{s}-unavailable', 'arm': CANDIDATE, 'seed': s,
                  'learning_rate': None, 'origin': 'unavailable'} for s in old.SEEDS]
    rows += inherited_identities()
    old.require(len(rows) == 46, 'all46 cost identities')
    return rows


def validate_rows(rows, cfg):
    roster = {r['key']: r for r in (*parent.identities(), *identities())}
    expected = {(n, k, h) for n in EXPOSED for k in roster for h in (64, 128)}
    ids = [(r['recording'], r['fit_key'], r['horizon']) for r in rows]
    old.require(len(ids) == len(set(ids)) == 488 and set(ids) == expected, 'all488 scheduled score rows')
    for r in rows:
        old.require(all(r[k] == roster[r['fit_key']][k] for k in ('arm', 'seed', 'learning_rate')), 'score identity joins')
        old.require(r['status'] in ('PASS', 'FAILED'), 'declared score status')
        if r['status'] == 'PASS':
            old.require(r['error'] is None and old.valid_metric_row(r) and r['metrics']['windows'] == 22
                        and r['metrics']['scalars'] == 22*r['horizon']*6 and r['metrics']['horizon'] == r['horizon'], 'complete metric geometry')
        else:
            old.require(r['error'] is not None and r['metrics'] is None, 'explicit failed metric')
    old.require(cfg == config(), 'immutable position-observer config')


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
        for arm in ELIGIBLE:
            subset = [r for r in rows if r['recording'] == name and r['arm'] == arm and r['horizon'] == 128 and selected(r, selection)]
            count = 1 if arm in REFERENCES else 3
            if len(subset) == count and all(old.valid_metric_row(r) for r in subset):
                values[arm] = float(np.mean([r['metrics']['standardized_rmse'] for r in subset]))
        details[name] = {'means': values}
    expected = {(r['key'], r['arm'], r['seed'], r['learning_rate']) for r in resource_identities(selection)}
    actual = [(r['key'], r['arm'], r['seed'], r['learning_rate']) for r in resources]
    old.require(len(actual) == len(set(actual)) and set(actual) == expected, 'all selected timing attempts, no omissions')
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
            if good:
                times.append(duration); sizes.append(sum(parts))
        costs[arm] = {'latency': float(np.median(times)), 'bytes': max(sizes)} if good else None
    frontier = all(a in means and costs[a] is not None for a in ELIGIBLE)
    complete = frontier and all(selection['selected_rates'][a] is not None for a in LEARNED)
    complete = complete and all(old.valid_metric_row(r) for r in rows if selected(r, selection))
    candidate = means.get(CANDIDATE); controls = [means.get(a) for a in ELIGIBLE if a != CANDIDATE]
    best = min(controls) if all(v is not None for v in controls) else None
    improvement = candidate is not None and best is not None and best > 0 and candidate <= .95*best
    no_harm = all(all(a in d['means'] for a in (*SIMPLE, CANDIDATE)) and d['means'][CANDIDATE]
                  <= 1.02*min(d['means'][a] for a in SIMPLE) for d in details.values())
    left, base = costs[CANDIDATE], costs['last_two']
    latency = left is not None and base is not None and left['latency'] <= 1.5*base['latency']
    dominators = []
    if frontier:
        point = (candidate, left['latency'], left['bytes'])
        for a in ELIGIBLE:
            if a == CANDIDATE: continue
            other = (means[a], costs[a]['latency'], costs[a]['bytes'])
            if all(x <= y for x, y in zip(other, point, strict=True)) and any(x < y for x, y in zip(other, point, strict=True)):
                dominators.append(a)
    conditions = [{'name': n, 'passed': bool(v)} for n, v in zip(CONDITIONS, (complete, improvement, no_harm, latency, frontier and not dominators), strict=True)]
    passed = sum(c['passed'] for c in conditions)
    return {'status': 'POSITION_OBSERVER_DEVELOPMENT_PASS' if passed == 5 else 'POSITION_OBSERVER_DEVELOPMENT_FAIL', 'passed': passed, 'total': 5,
            'conditions': conditions, 'details': details, 'equal_file_means': means, 'costs': costs,
            'best_control_mean': best, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'All four files exposed development; not confirmation, independent robots, novelty or control'}


def probe_schedule():
    return [
        {'key': f'{origin}-{seed}-batch0', 'seed': seed, 'batch_index': 0,
         'gain_origin': origin}
        for seed in (8101, 8102, 8103) for origin in ('identity', 'position')
    ] + [{'key': 'archived-8103-batch22', 'seed': 8103, 'batch_index': 22,
          'gain_origin': 'archived'}]


def infer_diagnostic(model, batch, prefix):
    """Same public inputs/casts as old.infer, with one prefix collector.

    Every probe uses PositionAdapter learned, including old [I;I] and archived
    gains loaded unchanged into it. Qualified same-gain parity applies; the
    recurrence is not duplicated or run a second time for diagnostics.
    """
    q = torch.from_numpy(np.asarray(batch['q_context'], dtype=np.float32))
    u = torch.from_numpy(np.asarray(batch['u_context'], dtype=np.float32))
    future = torch.from_numpy(np.asarray(batch['future_u'], dtype=np.float32))
    return model(future, model.condition(q, u, diagnostics=prefix))[0]


def native_norm_record(norm):
    """JSON-safe classification without converting nonfinite norms to success."""
    value = float(norm.detach())
    kind = ('finite' if math.isfinite(value) else 'nan' if math.isnan(value)
            else 'positive_infinity' if value > 0 else 'negative_infinity')
    return {'kind': kind, 'value': value if kind == 'finite' else None}


def gain_gradient_snapshot(model, frozen_sha):
    """Roster checks only, no finiteness guard, clipping or repair."""
    check_frozen(model, frozen_sha)
    parameters = dict(model.named_parameters())
    old.require({n for n, p in parameters.items() if p.requires_grad} == {'gain'}
                and {n for n, p in parameters.items() if p.grad is not None} == {'gain'},
                'only the 72-entry gain has a present training gradient')
    gain = parameters['gain'].grad
    old.require(gain.device.type == 'cpu' and gain.dtype == torch.float32
                and tuple(gain.shape) == (12, 6), 'raw float32 gain gradient schema')
    return gain.detach().cpu().numpy().copy()


def aggregate_diagnostics(records):
    """Pure saved-summary aggregation, not another gradient computation.

    backward_calls counts completed backward calls with captured gradients;
    an arbitrary failed backward is FATAL and may have no captured gradient.
    Prefix counts/maxima include any completed portion of a failed attempt.
    """
    gradients = [r['gradient'] for r in records if r['gradient'] is not None]
    native = [r['native_norm'] for r in records if r['native_norm'] is not None]
    result = {'attempts': len(records), 'backward_calls': len(gradients), 'clip_calls': len(native),
              'nonfinite_gradient_attempts': sum(not r['all_finite'] for r in gradients),
              'native_norm_nonfinite_attempts': sum(r['kind'] != 'finite' for r in native),
              'prefix_steps': sum(r['prefix'].get('prefix_steps', 0) for r in records)}
    for key in ('max_abs', 'norm64'):
        available = [r[key] for r in gradients if r[key] is not None]
        result['max_gradient_'+('abs' if key == 'max_abs' else key)] = max(available, default=None)
    for key in ('max_state_abs', 'max_state_norm64', 'max_innovation_abs', 'max_innovation_norm64'):
        result[key] = max((r['prefix'].get(key, 0.) for r in records), default=0.)
    return result


def probe_one(model, batch, identity, folder, check):
    """One no-update probe, including the unchanged native clipping operation."""
    old.require(identity in probe_schedule(), 'one of seven fixed FIT probes')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before, frozen_sha = confirmation.parameter_hash(model), cell_hash(model)
    model.zero_grad(set_to_none=True)
    prefix, arrays, gradient, native, loss_value = {}, {}, None, None, None
    status, error, outcome = 'PASS', None, None
    try:
        check()
        check_frozen(model, frozen_sha)
        prediction = infer_diagnostic(model, batch, prefix)
        target = torch.from_numpy(np.asarray(batch['target'], dtype=np.float32))
        loss = (prediction-target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise old.FitFailure('nonfinite autoregressive training loss')
        loss_value = float(loss.detach())
        loss.backward()
        arrays['gain_gradient'] = gain_gradient_snapshot(model, frozen_sha)
        gradient = gradient_diagnostics(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config()['gradient_clip'],
                                             error_if_nonfinite=False)
        arrays['native_norm'] = norm.detach().cpu().numpy().copy()
        native = native_norm_record(norm)
        if not bool(torch.isfinite(norm)):
            raise old.FitFailure('nonfinite training gradient norm')
        outcome = 'finite'
        check()
    except old.FitFailure as exc:
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
        outcome = ('nonfinite_gradient_entries' if gradient is not None and not gradient['all_finite']
                   else 'native_norm_overflow' if native is not None and native['kind'] != 'finite'
                   else 'forward_or_loss_failure')
    except BaseException as exc:
        status, error = 'FATAL', {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        after = confirmation.parameter_hash(model)
        unchanged = after == before
        # Clip mutates gradients, so clear them only after preserving PRE-clip
        # evidence. It must never have modified the cell or any parameter.
        check_frozen(model, frozen_sha)
        model.zero_grad(set_to_none=True)
        np.savez_compressed(folder/'preclip.npz', **arrays)
        receipt = {**identity, 'status': status, 'outcome': outcome, 'error': error,
                   'prefix': prefix, 'loss': loss_value, 'gradient': gradient,
                   'native_norm': native, 'parameter_hash_before': before,
                   'parameter_hash_after': after, 'parameters_unchanged': unchanged,
                   'frozen_cell_sha256': frozen_sha, 'seconds': time.monotonic()-started,
                   'optimizer_steps': 0, 'files': {'preclip.npz': old.descriptor(folder/'preclip.npz')},
                   'scope': 'One prescribed FIT batch, forward/backward/native clipping only; no optimizer or selection'}
        old.write_json(folder/'receipt.json', receipt)
        old.require(unchanged, 'no probe parameter mutation')
    return receipt


def train_one(model, records, batches, *, cfg, lr, folder, check=lambda: None, fit_started=None):
    """Original training path with observational diagnostics, no rescue guard."""
    started = time.monotonic() if fit_started is None else fit_started
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'initial.npz', **old.weights(model))
    frozen_sha = cell_hash(model)
    check_frozen(model, frozen_sha)
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
            last_gradient['gain_gradient'] = gain_gradient_snapshot(model, frozen_sha)
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
            check_frozen(model, frozen_sha)
            check()
            if time.monotonic()-started > cfg['fit_cap_seconds']:
                raise old.FitFailure('single-fit wall cap exceeded')
    except old.FitFailure as exc:
        status, error = 'FAILED', {'type': type(exc).__name__, 'message': str(exc)}
    except ValueError as exc:
        # Preserve the original trainer's one exact legacy numerical guard.
        # PositionAdapter already translates its own qualified numeric guards.
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


def run(registration, output):
    clock = SuspendClock(); start_ns, started = clock.now_ns(), time.monotonic()
    plan, sha = authenticate(registration); cfg = config(); output = Path(output)
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    fits, models, resources, states = [], {}, [], {}
    rows, attempts, probes = [], [], []
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
        parent_results = read(output/'parent/results.json')
        rows = parent_results['rows'].copy()
        attempts = read(output/'parent/prediction-attempts.json')
        old.require(len(rows) == 416 and len(attempts) == 208, 'complete unchanged parent outcomes')
        norm = load(plan, 'parent/normalizers.npz'); linear = load(plan, 'parent/linear.npz')['coefficient']
        physical_fit = [{'name': n, **load(plan, 'data/'+n)} for n in cfg['partitions']['fit']]
        fit_data = [old.normalized_record(r, norm) for r in physical_fit]
        batches = {s: load(plan, f'parent/batches-{s}.npz') for s in old.SEEDS}
        for seed, bank in batches.items():
            expected = old.make_batches([len(r['q']) for r in fit_data], seed, cfg)
            old.require(set(bank) == set(expected) and all(np.array_equal(bank[k], expected[k]) for k in bank), 'same paired4096 batches')
        cells = {s: load(plan, f'parent/last_two-{s}-fixed/final.npz') for s in old.SEEDS}
        def initialize(arm, seed):
            model = model_for(arm, seed, linear); arrays = cells[seed]
            old.require(set(arrays) == {'cell.'+n for n in model.cell.state_dict()}, 'selected exact cell schema')
            old.require(all(arrays['cell.'+n].shape == tuple(p.shape) and arrays['cell.'+n].dtype == np.float32
                            and np.isfinite(arrays['cell.'+n]).all() for n, p in model.cell.state_dict().items()), 'finite float32 cells')
            model.cell.load_state_dict({n: torch.from_numpy(arrays['cell.'+n]) for n in model.cell.state_dict()}, strict=True)
            digest = cell_hash(model); check_frozen(model, digest); return model, digest
        for identity in probe_schedule():
            check(); seed = identity['seed']; model, digest = initialize(CANDIDATE, seed)
            if identity['gain_origin'] == 'identity':
                with torch.no_grad(): model.gain.copy_(torch.cat((torch.eye(6), torch.eye(6))))
            elif identity['gain_origin'] == 'archived':
                arrays = load(plan, 'diagnostic/observer_learned-8103-lr0/final.npz')
                model.load_state_dict({n: torch.from_numpy(v) for n, v in arrays.items()}, strict=True)
            batch = old.window_batch(fit_data, {k: v[identity['batch_index']] for k,v in batches[seed].items()}, 32, 128)
            result = probe_one(model, batch, identity, output/'probes'/identity['key'], check)
            model.zero_grad(set_to_none=True); check_frozen(model, digest); probes.append(result)
            print(json.dumps({'probe': identity['key'], 'outcome': result['outcome']}), flush=True)
        old.write_json(output/'probes.json', probes)
        for identity in identities()[:6]:
            check(); before = time.monotonic(); fit_ns = clock.now_ns(); arm, seed, key = identity['arm'], identity['seed'], identity['key']
            model, digest = initialize(arm, seed)
            def fit_check(ns=fit_ns):
                check(); history.check_deadline(clock, ns, cfg['fit_cap_seconds'], whole=False)
            result = train_one(model, fit_data, batches[seed], cfg=cfg, lr=identity['learning_rate'],
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
        old.require(len(fits) == 6, 'all six new attempts before evaluation loads')
        for identity in identities()[6:]:
            check(); arm, seed, key = identity['arm'], identity['seed'], identity['key']
            model, digest = initialize(arm, seed); folder = output/key; folder.mkdir()
            np.savez_compressed(folder/'final.npz', **old.weights(model))
            model.eval(); models[key] = model; states[key] = confirmation.parameter_hash(model)
            fits.append({**identity, 'effective_status': 'PASS', 'frozen_cell_sha256': digest, 'resources': resource_model(model, arm)})
        parent_fits = {r['key']: r for r in read(output/'parent/fits.json')}
        for identity in inherited_identities():
            if identity['arm'] in REFERENCES: continue
            check(); arm, seed, key = identity['arm'], identity['seed'], identity['key']
            arrays = load(plan, 'parent/'+key+'/final.npz'); model = model_for(arm, seed, linear)
            model.load_state_dict({n: torch.from_numpy(v) for n,v in arrays.items()}, strict=True)
            inherited = parent_fits[key]
            old.require(inherited['effective_status'] == 'PASS', 'successful inherited control')
            folder = output/key; folder.mkdir(); (folder/'final.npz').write_bytes((output/'parent'/key/'final.npz').read_bytes())
            fits.append(inherited); model.eval(); models[key] = model; states[key] = confirmation.parameter_hash(model)
        old.require(len(models) == len(fits) == 45, 'all45 models before evaluation arrays')
        old.write_json(output/'checkpoint-barrier.json', {'fresh_fit_attempts': 6, 'fixed_models': 3, 'inherited_models': 36,
            'diagnostic_probes': 7, 'exposed_loads_this_run': 0, 'all_evaluation_data_exposed': True, 'state_sha256': states,
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
                attempts.append(attempt); old.write_json(output/f'completed-prediction-{len(attempts)-208:03d}.json', attempt)
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
            'monotonic_seconds': time.monotonic()-started, 'clock': clock.backend, 'fresh_fits': 6, 'fixed_models': 3, 'inherited_models': 36, 'diagnostic_probes': 7,
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
