"""Closed position-observer audit: independent metrics/rules and qualified replay.

Parent scores are authenticated and preserved without replaying parent banks.
No training, optimizer probes, raw MAT decoding, or scientific generation.
"""
from __future__ import annotations

import argparse
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

import audit_robot_observer_study as parent_audit

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-position-observer-audit-v1'
STUDY_VERSION = 'robot-position-observer-study-v1'
CANDIDATE = 'observer_position'
FIXED = 'observer_position_fixed'
SEEDS, RATES = parent_audit.SEEDS, parent_audit.RATES
REFS, THREADS, DEV, EXPOSED = parent_audit.REFS, parent_audit.THREADS, parent_audit.DEV, parent_audit.EXPOSED
PARENT_ARMS = tuple(a for a in parent_audit.ARMS if a != 'observer_learned')
ELIGIBLE = (CANDIDATE, FIXED, *PARENT_ARMS, *REFS)
DISPLAY = (CANDIDATE, FIXED, *parent_audit.ARMS, *REFS)
CONDITIONS = parent_audit.CONDITIONS
SOURCES = (*parent_audit.SOURCES, 'scripts/publish_robot_observer_study.py',
           'tests/test_publish_robot_observer_study.py', 'src/openjev/research/robot_position_observer.py',
           'tests/test_robot_position_observer.py', 'scripts/robot_position_observer_study.py',
           'tests/test_robot_position_observer_study.py', 'research/robot-position-observer-protocol.md',
           'scripts/audit_robot_position_observer.py', 'tests/test_audit_robot_position_observer.py')
LAUNCHER = 'scripts/launch_robot_position_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:], LAUNCHER)
STUDY = 'output/robot-position-observer-study-v1'
ENGINEERING = 'output/robot-position-observer-engineering-v1'
REGISTRATION = 'research/robot-position-observer-registration.json'
COMMAND = ['.venv/bin/python', '-u', 'scripts/robot_position_observer_study.py', '--registration', REGISTRATION, '--output', STUDY]
NUMERIC_ERRORS = parent_audit.NUMERIC_ERRORS
require, read, descriptor, pin = parent_audit.require, parent_audit.read, parent_audit.descriptor, parent_audit.pin
checked_pin, close, scored, metric_rows = parent_audit.checked_pin, parent_audit.close, parent_audit.scored, parent_audit.metric_rows
windows, state_hash = parent_audit.windows, parent_audit.state_hash


def identities():
    return ([{'key': f'{CANDIDATE}-{seed}-lr{i}', 'arm': CANDIDATE, 'seed': seed,
              'learning_rate': rate, 'origin': 'fresh'} for seed in SEEDS for i, rate in enumerate(RATES)]
            + [{'key': f'{FIXED}-{seed}-fixed', 'arm': FIXED, 'seed': seed,
                'learning_rate': None, 'origin': 'fixed'} for seed in SEEDS])


def inherited_identities(rates):
    require(set(rates) == set(PARENT_ARMS), 'complete inherited selected-rate roster')
    result = []
    for row in parent_audit.identities():
        arm = row['arm']
        if arm in REFS or (arm in PARENT_ARMS and row['learning_rate'] == rates[arm]):
            result.append(row)
    require(len(result) == 40, '36 inherited controls and4 references')
    return result


def storage(arm):
    if arm not in (CANDIDATE, FIXED):
        return parent_audit.storage(arm)
    spec = parent_audit.storage('observer_learned' if arm == CANDIDATE else 'observer_fixed')
    return spec


def decisions(rows, resources, cfg):
    """One new DEV2-selected recipe; every old recipe remains frozen."""
    rates = cfg['inherited_rates']
    inherited_identities(rates)
    lookup = {(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']): r for r in rows}
    recipe_roster = (*parent_audit.identities(), *identities())
    expected = {(name, r['arm'], r['seed'], r['learning_rate'], h)
                for name in EXPOSED for r in recipe_roster for h in (64, 128)}
    require(len(rows) == len(lookup) == 488 and set(lookup) == expected, 'complete488 duplicate-free metric roster')
    keys = {(r['arm'], r['seed'], r['learning_rate']): r['key'] for r in recipe_roster}
    require(all(r['fit_key'] == keys[r['arm'], r['seed'], r['learning_rate']] for r in rows), 'metric/checkpoint identity join')

    def metric(name, arm, seed, rate, horizon=128):
        row = lookup[name, arm, seed, rate, horizon]
        require(row['status'] in ('PASS', 'FAILED'), 'declared metric status')
        if row['status'] == 'FAILED':
            require(row['metrics'] is None and row['error'] is not None, 'retained metric failure')
            return None
        value = row['metrics']
        require(row['error'] is None and isinstance(value, dict), 'successful metric evidence')
        numbers = [value[k] for k in ('standardized_rmse', 'standardized_sse', 'physical_rmse_deg')]
        numbers += value['per_joint_rmse_deg']
        require(type(value['scalars']) is int and value['scalars'] == 22*horizon*6
                and value['windows'] == 22 and value['horizon'] == horizon and len(value['per_joint_rmse_deg']) == 6
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in numbers), 'finite metric geometry')
        return value

    for name, arm, seed, rate, h in expected:
        metric(name, arm, seed, rate, h)
    options, candidates = [], []
    for rate in RATES:
        values = [metric(name, CANDIDATE, seed, rate) for name in DEV for seed in SEEDS]
        sse = sum(v['standardized_sse'] for v in values) if all(v is not None for v in values) else None
        options.append({'learning_rate': rate, 'complete': sse is not None, 'pooled_sse': sse})
        if sse is not None:
            candidates.append((sse, rate))
    chosen = min(candidates)[1] if candidates else None
    selection = {'selected_rates': {CANDIDATE: chosen}, 'options': {CANDIDATE: options}, 'scope': cfg['selection_scope']}
    chosen_rates = {**rates, CANDIDATE: chosen, FIXED: None}
    details, means = {}, {}
    finite = chosen is not None
    for name in EXPOSED:
        values = {}
        for arm in ELIGIBLE:
            if arm == CANDIDATE and chosen is None:
                continue
            seeds, rate = ((None,), None) if arm in REFS else (SEEDS, chosen_rates[arm])
            subset = [metric(name, arm, seed, rate) for seed in seeds]
            finite &= all(metric(name, arm, seed, rate, h) is not None for seed in seeds for h in (64, 128))
            if all(v is not None for v in subset):
                values[arm] = sum(v['standardized_rmse'] for v in subset)/len(subset)
        details[name] = {'means': values}
    for arm in ELIGIBLE:
        if all(arm in details[name]['means'] for name in EXPOSED):
            means[arm] = sum(details[name]['means'][arm] for name in EXPOSED)/4
    cost_roster = resource_identities(selection, rates)
    require(len(resources) == 46, 'all46 resource attempts')
    for row, identity in zip(resources, cost_roster, strict=True):
        require(all(row[k] == v for k, v in identity.items()), 'ordered selected resource identity')
        require(row['status'] in ('PASS', 'FAILED', 'UNAVAILABLE'), 'declared resource status')
        require((row['error'] is None and isinstance(row['timing'], dict)) if row['status'] == 'PASS'
                else row['error'] is not None and row['timing'] is None, 'resource status and evidence agree')
    costs = {}
    for arm in ELIGIBLE:
        subset = [r for r in resources if r['arm'] == arm]
        require(len(subset) == (1 if arm in REFS else 3), 'complete family cost roster')
        costs[arm] = None
        if any(r['status'] != 'PASS' for r in subset):
            continue
        latency = [r['timing']['median_seconds'] for r in subset]
        sizes = [[r[k] for k in ('parameter_bytes', 'buffer_bytes', 'state_bytes', 'normalizer_bytes')] for r in subset]
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in latency)
                and all(type(v) is int and v >= 0 for parts in sizes for v in parts), 'finite complete request costs')
        costs[arm] = {'latency': sorted(latency)[len(latency)//2], 'bytes': max(sum(p) for p in sizes)}
    controls = tuple(a for a in ELIGIBLE if a != CANDIDATE)
    frontier = all(a in means and costs[a] is not None for a in ELIGIBLE)
    best = min(means[a] for a in controls) if all(a in means for a in controls) else None
    gain = CANDIDATE in means and best is not None and best > 0 and means[CANDIDATE] <= .95*best
    simple = ('last_two', 'local_affine', 'temporal_affine', 'observer_fixed', FIXED)
    harm = all(all(a in details[name]['means'] for a in (CANDIDATE, *simple))
               and details[name]['means'][CANDIDATE] <= 1.02*min(details[name]['means'][a] for a in simple)
               for name in EXPOSED)
    left, right = costs[CANDIDATE], costs['last_two']
    latency = left is not None and right is not None and left['latency'] <= 1.5*right['latency']
    dominators = []
    if frontier:
        point = means[CANDIDATE], left['latency'], left['bytes']
        for arm in controls:
            value = means[arm], costs[arm]['latency'], costs[arm]['bytes']
            if all(a <= b for a, b in zip(value, point, strict=True)) and any(a < b for a, b in zip(value, point, strict=True)):
                dominators.append(arm)
    flags = finite and frontier, gain, harm, latency, frontier and not dominators
    conditions = [{'name': n, 'passed': bool(v)} for n, v in zip(CONDITIONS, flags, strict=True)]
    passed = sum(r['passed'] for r in conditions)
    return {'selection': selection, 'result': {'status': 'POSITION_OBSERVER_DEVELOPMENT_PASS' if passed == 5 else 'POSITION_OBSERVER_DEVELOPMENT_FAIL',
            'passed': passed, 'total': 5, 'conditions': conditions, 'details': details, 'equal_file_means': means,
            'costs': costs, 'best_control_mean': best, 'frontier_complete': frontier, 'dominators': dominators,
            'scope': 'All four files exposed development; not confirmation, independent robots, novelty or control'}}


def resource_identities(selection, rates):
    chosen = selection['selected_rates'][CANDIDATE]
    selected = [r for r in identities() if r['arm'] == FIXED or r['learning_rate'] == chosen]
    if chosen is None:
        selected += [{'key': f'{CANDIDATE}-{seed}-unavailable', 'arm': CANDIDATE, 'seed': seed,
                      'learning_rate': None, 'origin': 'unavailable'} for seed in SEEDS]
    return [*selected, *inherited_identities(rates)]


def validate_frozen_evidence(initial, final, optimizer, backbone, mode, receipt, np, updates=4096):
    """The 590 inherited entries never change; only the 72-value gain owns Adam."""
    require(mode in (CANDIDATE, FIXED) and set(initial) == set(final) == {*backbone, 'gain'}, 'position checkpoint keys')
    require(sum(v.size for v in backbone.values()) == 590 and all(n.startswith('cell.') for n in backbone), '590 cell identity')
    for name, value in backbone.items():
        require(value.dtype == initial[name].dtype == final[name].dtype == np.float32
                and value.shape == initial[name].shape == final[name].shape and np.isfinite(value).all()
                and value.tobytes() == initial[name].tobytes() == final[name].tobytes(), 'immutable cell: '+name)
    desired = np.concatenate((np.eye(6, dtype=np.float32), np.zeros((6, 6), dtype=np.float32)))
    require(initial['gain'].dtype == final['gain'].dtype == np.float32 and initial['gain'].shape == final['gain'].shape == (12, 6)
            and np.array_equal(initial['gain'], desired), 'position-only gain initialization')
    steps = receipt['completed_updates']
    require(type(steps) is int and 0 <= steps <= updates and receipt['status'] in ('PASS', 'FAILED'), 'completed fit steps/status')
    keys = {'gain/'+s for s in ('step', 'exp_avg', 'exp_avg_sq')}
    require(set(optimizer) in (set(), keys), 'gain-only Adam state')
    if mode == FIXED:
        require(steps == 0 and not optimizer and np.array_equal(final['gain'], desired), 'fixed gain never updates')
    else:
        require(not steps or optimizer, 'completed steps retain Adam')
        if optimizer:
            for suffix in ('exp_avg', 'exp_avg_sq'):
                require(optimizer['gain/'+suffix].dtype == np.float32 and optimizer['gain/'+suffix].shape == (12, 6), 'Adam moment schema')
            v = optimizer['gain/step']
            require(v.dtype == np.float32 and v.shape == () and np.isfinite(v) and float(v) in (steps, steps+1), 'Adam attempted-step evidence')
    if receipt['status'] == 'PASS':
        require(steps == (updates if mode == CANDIDATE else 0)
                and all(np.isfinite(v).all() for v in (*final.values(), *optimizer.values())), 'finite complete fit')
        if mode == CANDIDATE:
            require(set(optimizer) == keys and float(optimizer['gain/step']) == updates, 'complete gain updates')


def gradient_summary(gradient, np):
    require(gradient.dtype == np.float32 and gradient.shape == (12, 6), 'raw preclip gain gradient')
    bad = int(np.count_nonzero(~np.isfinite(gradient)))
    maximum, norm = None, None
    if not bad:
        value = gradient.astype(np.float64)
        maximum = float(np.max(np.abs(value)))
        norm = 0. if maximum == 0 else maximum*math.sqrt(float(np.sum((value/maximum)**2)))
    return {'numel': 72, 'nonfinite_count': bad, 'max_abs': maximum, 'norm64': norm, 'all_finite': bad == 0}


def validate_parent_rows(rows, original):
    """Identity comparison only: never recompute inherited scores or selection."""
    retained = [r for r in rows if r['arm'] not in (CANDIDATE, FIXED)]
    require(len(original) == len(retained) == 416 and retained == original,
            'exact unchanged parent416 scalar rows in original order')


def validate_resources(resources, selection, rates, np):
    require(len(resources) == 46, 'all46 current cost slots')
    for row, identity in zip(resources, resource_identities(selection, rates), strict=True):
        spec = storage(identity['arm'])
        require(set(row) == set(identity) | set(spec) | {'status', 'error', 'timing'}
                and all(row[k] == v for k, v in identity.items()), 'resource schema and identity')
        close({k:row[k] for k in spec}, spec, 'all persistent and request storage')
        if identity['origin'] == 'unavailable':
            require(row['status'] == 'UNAVAILABLE' and row['timing'] is None
                    and row['error'] == {'type': 'UnavailableSelectedRecipe'}, 'explicit unavailable primary slot')
        elif row['status'] == 'FAILED':
            require(row['timing'] is None and row['error'] in [{'type': 'NonfiniteTiming', 'message': m}
                    for m in (*NUMERIC_ERRORS, 'finite complete timed request', 'nonfinite ridge prediction')], 'declared numerical timing failure')
        else:
            require(row['status'] == 'PASS' and row['error'] is None, 'successful timing identity')
            timing = row['timing']
            require(set(timing) == {'seconds', 'median_seconds', 'p95_seconds', 'scope'}
                    and timing['scope'] == parent_audit.prior_audit.TIMING_SCOPE
                    and len(timing['seconds']) == 20
                    and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in timing['seconds']),
                    'twenty original complete-request durations')
            close(timing['median_seconds'], float(np.median(timing['seconds'])), 'saved median')
            close(timing['p95_seconds'], float(np.percentile(timing['seconds'], 95)), 'saved p95')


def original_admission(study, run_receipt):
    """Fixed source, launch, qualification and opaque output joins before arrays."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    engineering, registration = ROOT/ENGINEERING, ROOT/REGISTRATION
    require(study == ROOT/STUDY and run_receipt == engineering/'run-process-01.json', 'fixed original study/process')
    plan, sha = read(registration), descriptor(registration)['sha256']
    require(plan['version'] == STUDY_VERSION and set(plan['sources']) == set(SOURCES), 'exact54 frozen sources')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT/name) == expected == descriptor(study/'sources'/name), 'source/snapshot: '+name)
    require(descriptor(study/'registration.json') == descriptor(registration)
            and descriptor(ROOT/LAUNCHER) == plan['launcher'], 'registration/launcher identity')
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread environment')
    process, launch = read(run_receipt), read(engineering/'run-launch-01.json')
    keys = {'command', 'prefit_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == keys and set(process) == keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'}
            and all(process[k] == v for k, v in launch.items()), 'original launch/terminal schema and join')
    require(process['command'] == COMMAND and process['registration_sha256'] == sha and process['launcher'] == plan['launcher']
            and process['thread_env'] == dict.fromkeys(THREADS, '1') and type(process['returncode']) is int
            and process['returncode'] == 0 and process['external_timeout'] is False
            and type(process['elapsed_seconds']) in (int, float) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= 7260 and descriptor(run_receipt.with_suffix('.log')) == process['log'],
            'successful bounded original process')
    commit = process['prefit_commit']
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'prefit commit identity')
    for name in (*SOURCES, LAUNCHER, REGISTRATION):
        require(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) == (ROOT/name).read_bytes(), 'prefit committed bytes: '+name)
    q = read(checked_pin(plan['qualification'], 'qualification')['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_position_observer.py',
                 'tests/test_robot_position_observer_study.py', 'tests/test_audit_robot_position_observer.py']]
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
            and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(THREADS, '1')
            and [r['command'] for r in q['commands']] == commands, 'exact original qualification')
    require(datetime.fromisoformat(q['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification and registration precede launch')
    for row in q['commands']:
        require(type(row['returncode']) is int and row['returncode'] == 0
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and 0 < row['seconds'] <= 180
                and descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification command/log')
    manifest, paths = read(study/'manifest.json'), list(study.rglob('*'))
    require(set(manifest) == {'files'} and not any(p.is_symlink() for p in paths), 'regular complete output inventory')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(manifest['files']) | {'manifest.json', 'receipt.json'},
            'no omitted output files')
    for name, expected in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and str(Path(name)) == name
                and descriptor(study/name) == expected, 'opaque output pin: '+name)
    runtime = read(study/'runtime.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform()
            and runtime['machine'] == platform.machine() and runtime['torch_threads'] == 1
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1'), 'same qualified runtime/platform')
    require(runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'), 'native suspend clock')
    for package in ('numpy', 'torch'):
        require(importlib.metadata.version(package) == runtime[package], 'same numeric package: '+package)
    inputs = {key: pin(path) for key, path in {'manifest': study/'manifest.json', 'producer_receipt': study/'receipt.json',
              'registration': registration, 'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'),
              'run_launch': engineering/'run-launch-01.json', 'runtime': study/'runtime.json', 'launcher': ROOT/LAUNCHER}.items()}
    inputs.update(qualification=plan['qualification'], qualification_logs=[pin(r['log']) for r in q['commands']])
    return plan, inputs


def parent_inputs(prior):
    """Exact metadata roles, including the one archived failed diagnostic state."""
    import publish_robot_observer_study as publisher
    folder, inventory = ROOT/publisher.STUDY, prior['inventory']
    expected = {k:v for k,v in prior['plan']['inputs'].items() if k.startswith('data/')}
    common = ('normalizers.npz', 'linear.npz', 'causal_ridge_1.npz', 'causal_ridge_1.json',
              'causal_ridge_100.npz', 'causal_ridge_100.json', *(f'batches-{s}.npz' for s in SEEDS))
    sources = {'parent/'+n:'parent/'+n for n in common}
    sources.update({'parent/'+n:n for n in ('results.json', 'fits.json', 'resources.json', 'prediction-attempts.json')})
    selected = prior['audit']['results']['selection']['selected_rates']
    require(selected == {'local_affine': .003, 'temporal_affine': .001, 'observer_learned': None}, 'exact frozen parent rate choices')
    rates = {**{a:selected[a] for a in parent_audit.LEARNED if a != 'observer_learned'},
             **dict.fromkeys(parent_audit.FIXED), **parent_audit.CACHED_RATES}
    for row in inherited_identities(rates):
        if row['arm'] not in REFS:
            name = row['key']+'/final.npz'
            sources['parent/'+name] = name
    failed = 'observer_learned-8103-lr0/final.npz'
    sources['diagnostic/'+failed] = failed
    for target, original in sources.items():
        # The parent itself inherited a parent/fits.json. That older ledger
        # must never shadow this study's immediate parent fit ledger.
        item = prior['plan']['inputs'][target] if target in {'parent/'+n for n in common} else pin(folder/original)
        require({k:item[k] for k in ('sha256','bytes')} == inventory[original], 'parent manifest role: '+original)
        expected[target] = item
    require(len(expected) == 61 and len(sources) == 50, '11 data plus50 inherited evidence inputs')
    return expected, rates


def native_norm_summary(value, np):
    require(value.dtype == np.float32 and value.shape == (), 'raw native float32 clipping norm')
    number = float(value)
    if math.isnan(number):
        return {'kind': 'nan', 'value': None}
    if math.isinf(number):
        return {'kind': 'positive_infinity' if number > 0 else 'negative_infinity', 'value': None}
    require(number >= 0, 'nonnegative finite native norm')
    return {'kind': 'finite', 'value': number}


def gradient_outcome(summary, native):
    if native is None:
        return 'forward_or_loss_failure'
    if not summary['all_finite']:
        return 'nonfinite_gradient_entries'
    return 'finite' if native['kind'] == 'finite' else 'native_norm_overflow'


def expected_config(parent, rates):
    cfg = dict(parent)
    for name in ('cached', 'cached_rates'):
        del cfg[name]
    cfg.update(version=STUDY_VERSION, learned=[CANDIDATE], fixed=[FIXED], families=list(DISPLAY),
               eligible_families=list(ELIGIBLE), inherited_rates=rates, wall_cap_seconds=7200.,
               diagnostic_probes=7, inherited_rows=416, inherited_prediction_attempts=208,
               new_fits=6, new_fixed_models=3, input_count=61,
               gradient_clipping='unchanged native float32 norm; float64 diagnostic only')
    return cfg


def authenticate(study, run_receipt):
    plan, inputs = original_admission(study, run_receipt)
    study = Path(study).resolve()
    import publish_robot_observer_study as publisher
    prior = publisher.authenticate(ROOT/publisher.STUDY, ROOT/publisher.AUDIT, ROOT/publisher.ENGINEERING)
    require(prior['audit']['results']['result']['status'] == 'OBSERVER_DEVELOPMENT_FAIL'
            and prior['audit']['results']['result']['passed'] == 0, 'closed failed parent remains failed')
    expected, rates = parent_inputs(prior)
    require(plan['inputs'] == expected and plan['config'] == expected_config(prior['plan']['config'], rates),
            'literal new config and61 admitted input roles')
    for name, item in expected.items():
        checked_pin(item, name)
        if not name.startswith('data/'):
            require(descriptor(study/name) == {k:item[k] for k in ('bytes','sha256')}, 'unchanged inherited byte copy: '+name)
    require(plan['parent_publication_manifest'] == pin(ROOT/publisher.OUTPUT/'manifest.json')
            and plan['parent_audit'] == pin(ROOT/publisher.AUDIT), 'closed parent audit and publication')
    receipt, runtime, process = read(study/'receipt.json'), read(study/'runtime.json'), read(run_receipt)
    fields = ('fresh_fits','fixed_models','inherited_models','diagnostic_probes','rows','prediction_attempts',
              'timing_attempts','raw_mat_decodes','saved_fit_loads','saved_exposed_loads')
    require(receipt['status'] == 'PASS' and receipt['registration_sha256'] == inputs['registration']['sha256']
            and [receipt[k] for k in fields] == [6,3,36,7,488,244,46,0,7,4]
            and receipt['all_evaluation_data_exposed'] is True and receipt['official_test_access'] is False
            and receipt['clock'] == runtime['clock'] and 0 < receipt['seconds'] <= 7200
            and 0 < receipt['monotonic_seconds'] <= process['elapsed_seconds'], 'complete bounded child terminal')
    inputs.update(parent_admission=prior['inputs'], parent_publication_inputs=prior['publication_inputs'],
                  parent_inputs=plan['inputs'], parent_audit=plan['parent_audit'],
                  parent_publication_manifest=plan['parent_publication_manifest'])
    return plan, inputs


DIAGNOSTICS_SCOPE = ('Per-attempt detached prefix maxima and PRE-clipping gradient summaries included in loop time; '
                     'only the last attempt retains raw gradient/native norm arrays; no historical backward replay')
PREFIX_KEYS = ('prefix_steps', 'max_state_abs', 'max_state_norm64', 'max_innovation_abs', 'max_innovation_norm64')
PROBE_SCOPE = 'One prescribed FIT batch, forward/backward/native clipping only; no optimizer or selection'


def validate_fit_receipt(receipt, trace, rate, cap):
    """Inherited optimizer semantics, with seven pinned observational payloads."""
    require(receipt.get('diagnostics_scope') == DIAGNOSTICS_SCOPE, 'declared diagnostic scope')
    require(set(receipt['files']) == {'initial.npz', 'final.npz', 'optimizer.npz', 'trace.json',
                                    'diagnostics.json', 'diagnostic-summary.json', 'last-gradient.npz'}, 'seven fit payloads')
    base = {k:v for k,v in receipt.items() if k != 'diagnostics_scope'}
    base['files'] = {k:receipt['files'][k] for k in ('initial.npz', 'final.npz', 'optimizer.npz', 'trace.json')}
    if base['error'] == {'type':'ValueError', 'message':'nonfinite joint-coupling output; no clipping or repair'}:
        base['error'] = {'type':'FitFailure', 'message':base['error']['message']}
    parent_audit.validate_fit_receipt(base, trace, rate, cap)


def validate_prefix(prefix, complete=False):
    require(isinstance(prefix, dict) and (not prefix or set(prefix) == set(PREFIX_KEYS)), 'five detached prefix summaries')
    if prefix:
        require(type(prefix['prefix_steps']) is int and 0 <= prefix['prefix_steps'] <= 30
                and all(type(prefix[k]) in (int,float) and math.isfinite(prefix[k]) and prefix[k] >= 0
                        for k in PREFIX_KEYS[1:]), 'finite prefix diagnostics')
    require(not complete or (bool(prefix) and prefix['prefix_steps'] == 30), 'complete thirty-correction prefix')


def validate_summary(summary, native):
    if summary is None:
        require(native is None, 'no clip without captured gradient')
        return
    require(set(summary) == {'numel','nonfinite_count','max_abs','norm64','all_finite'} and summary['numel'] == 72
            and type(summary['nonfinite_count']) is int and 0 <= summary['nonfinite_count'] <= 72
            and type(summary['all_finite']) is bool and summary['all_finite'] == (summary['nonfinite_count'] == 0),
            'gain-only gradient summary')
    if summary['all_finite']:
        require(all(type(summary[k]) in (int,float) and math.isfinite(summary[k]) and summary[k] >= 0
                    for k in ('max_abs','norm64')), 'finite float64 gain magnitudes')
    else:
        require(summary['max_abs'] is None and summary['norm64'] is None, 'nonfinite entries have no substituted magnitude')
    require(isinstance(native, dict) and set(native) == {'kind','value'} and native['kind'] in
            ('finite','nan','positive_infinity','negative_infinity'), 'native clipping norm classification')
    require((type(native['value']) in (int,float) and math.isfinite(native['value']) and native['value'] >= 0)
            if native['kind'] == 'finite' else native['value'] is None, 'JSON-safe native norm without repair')


def validate_raw_gradient(raw, summary, native, np):
    require(set(raw) == (set() if summary is None else {'gain_gradient','native_norm'}), 'exact raw preclip evidence roster')
    if summary is not None:
        close(summary, gradient_summary(raw['gain_gradient'], np), 'independent raw gain magnitudes')
        require(native == native_norm_summary(raw['native_norm'], np), 'raw native norm classification')


def aggregate_diagnostics(records):
    """Reconcile saved scalar summaries, not a second backward calculation."""
    gradients = [r['gradient'] for r in records if r['gradient'] is not None]
    native = [r['native_norm'] for r in records if r['native_norm'] is not None]
    result = {'attempts':len(records), 'backward_calls':len(gradients), 'clip_calls':len(native),
              'nonfinite_gradient_attempts':sum(r['nonfinite_count'] > 0 for r in gradients),
              'native_norm_nonfinite_attempts':sum(r['kind'] != 'finite' for r in native),
              'prefix_steps':sum(r['prefix'].get('prefix_steps',0) for r in records)}
    result['max_gradient_abs'] = max((r['max_abs'] for r in gradients if r['max_abs'] is not None), default=None)
    result['max_gradient_norm64'] = max((r['norm64'] for r in gradients if r['norm64'] is not None), default=None)
    for key in PREFIX_KEYS[1:]:
        result[key] = max((r['prefix'].get(key,0.) for r in records), default=0.)
    return result


def validate_diagnostics(records, summary, raw, receipt, trace, np):
    steps = receipt['completed_updates']
    require(isinstance(records,list) and 1 <= len(records) <= 4096 and len(records) in (steps,steps+1),
            'one diagnostic per attempted update')
    for i,row in enumerate(records,1):
        require(set(row) == {'update','prefix','gradient','native_norm','status','error'} and row['update'] == i
                and row['status'] in ('PASS','FAILED'), 'ordered attempt diagnostic schema')
        validate_prefix(row['prefix'], complete=row['gradient'] is not None)
        validate_summary(row['gradient'],row['native_norm'])
        if row['status'] == 'FAILED':
            require(i == len(records) and receipt['status'] == 'FAILED' and row['error'] == receipt['error'],
                    'only final preserved failed attempt')
        else:
            require(row['error'] is None and row['gradient'] is not None and row['gradient']['all_finite']
                    and row['native_norm']['kind'] == 'finite', 'completed finite attempted update')
        if i <= steps:
            require(row['gradient'] is not None and row['gradient']['all_finite'] and row['native_norm']['kind'] == 'finite',
                    'completed optimizer update has finite preclip evidence')
            close(row['native_norm']['value'], trace[i-1]['gradient_norm_before_clip'], 'native norm joins original trace')
    if receipt['status'] == 'PASS':
        require(len(records) == steps == 4096 and all(r['status'] == 'PASS' for r in records), 'successful complete diagnostics')
    validate_raw_gradient(raw, records[-1]['gradient'], records[-1]['native_norm'], np)
    close(summary, aggregate_diagnostics(records), 'all twelve diagnostic aggregate fields')


def probe_schedule():
    return [{'key':f'{kind}-{seed}-batch0','seed':seed,'batch_index':0,'gain_origin':kind}
            for seed in SEEDS for kind in ('identity','position')] + [
            {'key':'archived-8103-batch22','seed':8103,'batch_index':22,'gain_origin':'archived'}]


def validate_probe(receipt, raw, identity, parameter_digest, cell_digest, np):
    fields = {'status','outcome','error','prefix','loss','gradient','native_norm','parameter_hash_before',
              'parameter_hash_after','parameters_unchanged','frozen_cell_sha256','seconds','optimizer_steps','files','scope'}
    require(set(receipt) == set(identity)|fields and all(receipt[k] == v for k,v in identity.items()), 'fixed seven-probe identity')
    require(receipt['parameter_hash_before'] == receipt['parameter_hash_after'] == parameter_digest
            and receipt['parameters_unchanged'] is True and receipt['frozen_cell_sha256'] == cell_digest
            and type(receipt['optimizer_steps']) is int and receipt['optimizer_steps'] == 0
            and receipt['scope'] == PROBE_SCOPE and type(receipt['seconds']) in (int,float)
            and math.isfinite(receipt['seconds']) and receipt['seconds'] > 0, 'no-update probe evidence')
    require(receipt['status'] in ('PASS','FAILED') and set(receipt['files']) == {'preclip.npz'}, 'retained probe status/file')
    validate_prefix(receipt['prefix'], complete=receipt['gradient'] is not None)
    validate_summary(receipt['gradient'], receipt['native_norm'])
    validate_raw_gradient(raw,receipt['gradient'],receipt['native_norm'],np)
    require(receipt['loss'] is None or (type(receipt['loss']) in (int,float) and math.isfinite(receipt['loss'])
            and receipt['loss'] >= 0), 'finite or absent original probe loss')
    require(receipt['gradient'] is None or receipt['loss'] is not None, 'backward follows finite loss')
    if receipt['status'] == 'PASS':
        require(receipt['error'] is None and receipt['gradient'] is not None and receipt['gradient']['all_finite']
                and receipt['native_norm']['kind'] == 'finite' and receipt['outcome'] == 'finite', 'successful native probe')
    else:
        require(receipt['error'] in [{'type':'FitFailure','message':m} for m in
                (*NUMERIC_ERRORS,'nonfinite autoregressive training loss','nonfinite training gradient norm')], 'known probe numeric failure')
        require(receipt['outcome'] == gradient_outcome(receipt['gradient'],receipt['native_norm']), 'raw diagnostic failure classification')


def audit(study, run_receipt):
    started = time.monotonic()
    study = Path(study).resolve()
    plan, inputs = authenticate(study, run_receipt)
    import numpy as np
    import robot_position_observer_study as replay
    import torch
    torch.set_num_threads(1)
    cfg, roster = plan['config'], identities()
    counts = {'npz_decodes':0,'array_decodes':0,'qualified_model_replays':0,'parent_forecast_replays':0,
              'raw_mat_decodes':0,'official_test_decodes':0,'optimizer_updates':0,'backward_replays':0,
              'native_clip_replays':0,'timing_replays':0}
    expected = {'registration.json','runtime.json','fits.json','checkpoint-barrier.json','parameter-checks.json',
                'prediction-attempts.json','resources.json','results.json','probes.json'}
    expected |= {'sources/'+name for name in SOURCES}
    expected |= {name for name in plan['inputs'] if not name.startswith('data/')}

    def load(name, external=False):
        if external:
            require(name in plan['inputs'] and name.startswith('data/'), 'only11 saved physical recordings')
            path = checked_pin(plan['inputs'][name],name)['path']
        else:
            expected.add(name); path = study/name
        with np.load(path,allow_pickle=False) as bank:
            values = {name:bank[name].copy(order='K') for name in bank.files}
        counts['npz_decodes'] += 1; counts['array_decodes'] += len(values)
        return values

    def equal(actual,wanted,label):
        require(actual.dtype == wanted.dtype and actual.shape == wanted.shape
                and np.array_equal(actual,wanted,equal_nan=True),label)

    norm = load('parent/normalizers.npz')
    require(set(norm) == {'q_mean','q_std','u_mean','u_std'} and all(v.dtype == np.float64 and v.shape == (6,)
            and np.isfinite(v).all() for v in norm.values()) and all((norm[k]>0).all() for k in ('q_std','u_std')),
            'finite inherited normalization')
    fit_data = [load('data/'+name,external=True) for name in cfg['partitions']['fit']]
    require(len(fit_data) == 7,'seven inherited FIT recordings')
    for data in fit_data:
        windows(data,norm,np)
    for field in ('q','u'):
        values = np.concatenate([data[field][64:] for data in fit_data])
        equal(norm[field+'_mean'],values.mean(0),'unchanged FIT mean')
        equal(norm[field+'_std'],values.std(0),'unchanged FIT population scale')
    for seed in SEEDS:
        bank = load(f'parent/batches-{seed}.npz')
        require(set(bank) == {'record','start'},'saved batch schema')
        rng = np.random.Generator(np.random.PCG64(seed+520000))
        records = rng.integers(0,7,size=(4096,16),dtype=np.int64)
        starts = np.empty_like(records)
        for index in np.ndindex(records.shape):
            starts[index] = rng.integers(64,3636-32-128+1)
        equal(bank['record'],records,'paired original recording draws')
        equal(bank['start'],starts,'paired original window draws')
    linear_bank = load('parent/linear.npz')
    require(set(linear_bank) == {'coefficient'},'linear coefficient bank')
    linear = linear_bank['coefficient']
    require(linear.shape == (6,25) and linear.dtype == np.float64 and np.isfinite(linear).all(),'finite inherited linear coefficients')
    # Reference forecasts are retained from the admitted parent, not replayed.
    # The two reference NPZ files stay opaque; their complete contents are pinned.
    backbones = {s:load(f'parent/last_two-{s}-fixed/final.npz') for s in SEEDS}
    parent_rows = read(study/'parent/results.json')['rows']
    parent_attempts = read(study/'parent/prediction-attempts.json')
    parent_fits = read(study/'parent/fits.json')
    require(len(parent_rows) == 416 and len(parent_attempts) == 208
            and len(parent_fits) == len({r['key'] for r in parent_fits}) == 48,'complete immediate parent evidence')
    ledger = {r['key']:r for r in parent_fits}
    inherited = [r for r in inherited_identities(cfg['inherited_rates']) if r['arm'] not in REFS]
    fits = read(study/'fits.json')
    require(len(fits) == 45,'all9 new and36 inherited models')
    models, states, templates, passed = {},{},{},set()
    for number,(fit,identity) in enumerate(zip(fits,[*roster,*inherited],strict=True),1):
        key,arm,seed = identity['key'],identity['arm'],identity['seed']
        require(all(fit[k] == v for k,v in identity.items()),'ordered model identity')
        final = load(key+'/final.npz')
        if arm in (CANDIDATE,FIXED):
            model = replay.model_for(arm,seed,linear)
            template = model.state_dict(); backbone = backbones[seed]
            require(set(final) == set(template) and all(final[n].shape == tuple(v.shape) and final[n].dtype == np.float32
                    for n,v in template.items()),'qualified new checkpoint schema')
            require(set(backbone) == {'cell.'+n for n in model.cell.state_dict()},'exact inherited590 cell keys')
            model.cell.load_state_dict({n:torch.from_numpy(backbone['cell.'+n]) for n in model.cell.state_dict()},strict=True)
            constructed = {n:v.detach().numpy().copy(order='K') for n,v in model.state_dict().items()}
            digest = state_hash({n.removeprefix('cell.'):v for n,v in backbone.items()})
            require(fit['frozen_cell_sha256'] == digest,'independent immutable cell hash')
            templates[seed] = constructed
            if arm == CANDIDATE:
                expected |= {f'completed-fit-{number:02d}.json',key+'/trace.json',key+'/fit-receipt.json',
                             key+'/diagnostics.json',key+'/diagnostic-summary.json'}
                require(read(study/f'completed-fit-{number:02d}.json') == fit,'every fresh attempt retained')
                initial,optimizer = load(key+'/initial.npz'),load(key+'/optimizer.npz')
                receipt,trace = read(study/key/'fit-receipt.json'),read(study/key/'trace.json')
                require(receipt == fit['fit'],'original helper receipt joins fit ledger')
                validate_fit_receipt(receipt,trace,identity['learning_rate'],cfg['fit_cap_seconds'])
                for name,item in receipt['files'].items():
                    require(descriptor(study/key/name) == item,'all seven fit payload pins')
                require(set(initial) == set(constructed),'complete initial keys')
                for name in initial:
                    equal(initial[name],constructed[name],'exact inherited cell and[I;0] initialization')
                validate_frozen_evidence(initial,final,optimizer,backbone,arm,receipt,np)
                validate_diagnostics(read(study/key/'diagnostics.json'),read(study/key/'diagnostic-summary.json'),
                                     load(key+'/last-gradient.npz'),receipt,trace,np)
                require(type(fit['native_fit_seconds']) in (float,int) and math.isfinite(fit['native_fit_seconds'])
                        and fit['native_fit_seconds']>0 and fit['effective_status'] ==
                        ('FAILED' if fit['native_fit_seconds']>=1800 else receipt['status']),'native preservation deadline status')
                fields = set(identity)|{'fit','effective_status','native_fit_seconds','frozen_cell_sha256','resources'}
            else:
                for name,value in constructed.items():
                    equal(final[name],value,'fixed[I;0] state unchanged')
                validate_frozen_evidence(constructed,final,{},backbone,arm,{'status':'PASS','completed_updates':0},np)
                require(fit['effective_status'] == 'PASS','fixed model status')
                fields = set(identity)|{'effective_status','frozen_cell_sha256','resources'}
            require(set(fit) == fields,'new model metadata schema')
            if fit['effective_status'] == 'PASS':
                require(all(np.isfinite(v).all() for v in final.values()),'finite successful final state')
                model.load_state_dict({n:torch.from_numpy(v) for n,v in final.items()},strict=True)
                model.eval(); models[key] = model; passed.add(key)
                require(all(p.grad is None for p in model.parameters()),'no retained inference gradients')
        else:
            require(fit == ledger[key] and fit['effective_status'] == 'PASS','unchanged complete parent fit record')
            require(descriptor(study/key/'final.npz') == descriptor(study/'parent'/key/'final.npz'), 'byte-identical inherited checkpoint')
            require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in final.values()),'finite inherited model state')
        close(fit['resources'],storage(arm),'full model storage')
        states[key] = state_hash(final)
    barrier = {'fresh_fit_attempts':6,'fixed_models':3,'inherited_models':36,'diagnostic_probes':7,
               'exposed_loads_this_run':0,'all_evaluation_data_exposed':True,'state_sha256':states,
               'checkpoints':{r['key']:descriptor(study/r['key']/'final.npz') for r in fits}}
    require(read(study/'checkpoint-barrier.json') == barrier,'all45 model closure before exposed decoding')
    require(read(study/'parameter-checks.json') == {'before':states,'after':states,'unchanged':True},'immutable inference receipts')
    archived = load('diagnostic/observer_learned-8103-lr0/final.npz')
    require(set(archived) == set(templates[8103]),'archived failed probe checkpoint schema')
    for name,value in templates[8103].items():
        require(archived[name].dtype == value.dtype and archived[name].shape == value.shape,'archived checkpoint geometry')
        if name != 'gain': equal(archived[name],value,'archived cell identity')
    probes = read(study/'probes.json')
    require(len(probes) == 7,'seven no-update diagnostic receipts')
    for row,identity in zip(probes,probe_schedule(),strict=True):
        state = {n:v.copy(order='K') for n,v in templates[identity['seed']].items()}
        if identity['gain_origin'] == 'identity': state['gain'] = np.tile(np.eye(6,dtype=np.float32),(2,1))
        elif identity['gain_origin'] == 'archived': state = archived
        prefix = 'probes/'+identity['key']; expected.add(prefix+'/receipt.json')
        require(read(study/prefix/'receipt.json') == row,'original probe receipt join')
        require(row['files']['preclip.npz'] == descriptor(study/prefix/'preclip.npz'),'raw preclip byte pin')
        digest = state_hash({n.removeprefix('cell.'):v for n,v in backbones[identity['seed']].items()})
        validate_probe(row,load(prefix+'/preclip.npz'),identity,state_hash(state),digest,np)
    rows,attempts = parent_rows.copy(),parent_attempts.copy()
    fit_lookup = {r['key']:r for r in fits}
    for name in EXPOSED:
        starts,batch,target = windows(load('data/'+name,external=True),norm,np)
        bank = load('dev-windows-'+name+'.npz')
        require(set(bank) == {'starts','target'},'saved target window fields')
        equal(bank['starts'],starts,'all22 prescribed windows')
        equal(bank['target'],target,'independent u31-to-q32 target alignment')
        for identity in roster:
            key,arm = identity['key'],identity['arm']
            common = {'recording':name,'arm':arm,'seed':identity['seed'],'learning_rate':identity['learning_rate'],'fit_key':key}
            prediction,error = None,None
            filename = 'prediction-'+name+'-'+key+'.npz'
            if key not in passed:
                error = {'type':'FailedTrainingAttempt','effective_status':fit_lookup[key]['effective_status']}
            else:
                try:
                    counts['qualified_model_replays'] += 1
                    with torch.no_grad(): generated = replay.old.infer(models[key],batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) in NUMERIC_ERRORS,'known qualified numerical failure')
                    error = {'type':'NonfiniteEvaluation','message':str(exc)}
                else:
                    bank = load(filename); require(set(bank) == {'prediction'},'single forecast bank field')
                    prediction = bank['prediction']; equal(prediction,generated,'exact qualified new forecast replay')
            if prediction is None: require(not (study/filename).exists(),'failed forecast has no hidden bank')
            group = metric_rows(common,prediction,target,norm['q_std'],error,np); rows.extend(group)
            attempt = {**common,'status':'PASS' if all(r['status']=='PASS' for r in group) else 'FAILED',
                       'errors':[r['error'] for r in group],'prediction_file':filename if prediction is not None else None}
            attempts.append(attempt)
            completed = f'completed-prediction-{len(attempts)-208:03d}.json'; expected.add(completed)
            close(read(study/completed),attempt,'every new prediction attempt')
    saved_results = read(study/'results.json')
    require(saved_results['rows'][:416] == parent_rows and len(saved_results['rows']) == 488,'exact inherited416 row prefix')
    validate_parent_rows(saved_results['rows'],parent_rows)
    require(read(study/'prediction-attempts.json')[:208] == parent_attempts,'exact inherited208 attempt prefix')
    close(read(study/'prediction-attempts.json'),attempts,'all244 retained attempts')
    resources = read(study/'resources.json'); decision = decisions(rows,resources,cfg)
    validate_resources(resources,decision['selection'],cfg['inherited_rates'],np)
    for i,row in enumerate(resources,1):
        name = f'completed-timing-{i:02d}.json'; expected.add(name)
        require(read(study/name) == row,'all46 original timing receipts')
    results = {'version':cfg['version'],'config':cfg,'selection':decision['selection'],'rows':rows,'result':decision['result']}
    close(saved_results,results,'independent new metrics, fixed old recipes and five rules')
    require(read(study/'receipt.json')['scientific_result'] == decision['result']['status'],'terminal scientific result')
    require(set(read(study/'manifest.json')['files']) == expected,'exact dynamic output inventory')
    require(all(state_hash({n:v.detach().numpy() for n,v in m.state_dict().items()}) == states[k] for k,m in models.items()),
            'replay did not mutate current model state')
    require(authenticate(study,run_receipt) == (plan,inputs),'original evidence unchanged after audit')
    counts.update(fresh_fits=6,fixed_models=3,inherited_models=36,final_checkpoints=45,initial_checkpoints=6,
                  optimizer_checkpoints=6,diagnostic_probes=7,raw_gradient_banks=13,metric_rows=488,
                  inherited_metric_rows=416,new_metric_rows=72,prediction_attempts=244,inherited_prediction_attempts=208,
                  new_prediction_attempts=36,new_prediction_files=sum(r['prediction_file'] is not None for r in attempts[208:]),
                  resource_rows=46,requested_fresh_updates=24576,completed_fresh_updates=sum(r['fit']['completed_updates'] for r in fits[:6]),
                  failed_fresh_fits=sum(r['effective_status']!='PASS' for r in fits[:6]),
                  failed_metric_rows=sum(r['status']!='PASS' for r in rows),manifest_files=len(expected),
                  frozen_cell_checks=9,saved_fit_recordings=7,saved_exposed_recordings=4,condition_rows=5)
    return {'version':VERSION,'status':'PASS','agreement':True,'study':str(study),
            'registration_sha256':inputs['registration']['sha256'],'source_pins':plan['sources'],'auditor':pin(__file__),
            'inputs':inputs,'results':results,'resources':resources,'prediction_attempts':attempts,'probes':probes,
            'fit_diagnostics':{r['key']:read(study/r['key']/'diagnostic-summary.json') for r in fits[:6]},
            'counts':counts,'seconds':time.monotonic()-started,
            'checks':{'original_closed_process':True,'independent_new_metrics_and_rules':True,'parent_rows_retained_exactly':True,
                      'selection_original_dev2_only':True,'all590_backbone_values_immutable':True,'gain_only_adam_state':True,
                      'exact_qualified_new_model_replay':True,'raw_preclip_gradient_reconciliation':True,
                      'all_failed_attempts_retained':True},
            'scope':['No raw MAT numerical decoding, fitting, backward, clipping, optimizer or timing replay.',
                     'Parent416 scores and208 attempts are authenticated unchanged; their forecast banks are never decoded or replayed.',
                     'New recurrence replay uses qualified model code; new metrics, selection, five rules and gradient summaries are independent.',
                     'Training histories, prefix maxima and probe executions are source/receipt attestations; only seven probes and six last-attempt gradient arrays are numerically reconciled.',
                     'Historical media are hashed opaquely in parent admission. All four evaluation recordings are exposed development.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--run-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.parent.exists(),'exclusive audit output directory')
    result = audit(args.study,args.run_receipt)
    args.output.parent.mkdir(parents=True,exist_ok=False)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')
    with (args.output.parent/'manifest.json').open('x') as handle:
        json.dump({'files':{args.output.name:descriptor(args.output)}},handle,indent=2,sort_keys=True); handle.write('\n')
    print(json.dumps({'status':result['status'],'agreement':result['agreement'],'counts':result['counts'],
                      'scientific_status':result['results']['result']['status']}),flush=True)


if __name__ == '__main__':
    main()
