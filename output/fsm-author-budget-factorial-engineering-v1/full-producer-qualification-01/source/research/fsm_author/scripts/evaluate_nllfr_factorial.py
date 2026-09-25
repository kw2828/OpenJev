# SPDX-License-Identifier: GPL-3.0-or-later
"""Four fixed checkpoint/context-budget conditions using audited request caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import evaluate_nllfr as prior
import numpy as np

from openjev_fsm_author.benchmark import DEV_IDS, require
from openjev_fsm_author.nllfr import NAMES, physical_rollout, validate
from openjev_fsm_author.nllfr_context_budget import condition

ROOT, ENV = prior.ROOT, prior.ENV
pin, sha, write, event = prior.pin, prior.sha, prior.write, prior.event
VERSION = 'fsm-author-nllfr-factorial-v1'
CELLS = ('old16', 'old64', 'new16', 'new64')
STARTS = tuple(range(0, 7937, 256))
EXPERIMENT = {'cells': list(CELLS), 'context': 100, 'horizon': 128, 'records': 12,
    'requests_per_record': 32, 'forecast_attempts': 1536, 'record_slots': 48,
    'warmup_requests': 4, 'timed_requests': 96, 'outer_timeout_seconds': 3600,
    'rss_cap_bytes': 32*1024**3, 'scoring': 'cached standardized targets'}
REQUIRED = {'runtime_preflight', 'old_audit', 'old_audit_process', 'old_registration',
    'old_process', 'old_evaluation_process', 'new_registration', 'new_process', 'new_audit',
    'new_audit_process', 'new_audit_freeze', 'old_final', 'new_final', 'common_normalizer',
    'producer_qualification', 'source_review'}
MIN_SOURCES = {'research/fsm_author/scripts/evaluate_nllfr_factorial.py',
    'research/fsm_author/scripts/run_nllfr_factorial.py',
    'research/fsm_author/tests/test_nllfr_factorial.py',
    'research/fsm_author/src/openjev_fsm_author/nllfr_context_budget.py',
    'research/fsm_author/src/openjev_fsm_author/nllfr_context.py',
    'research/fsm_author/src/openjev_fsm_author/nllfr.py',
    'research/fsm_author/scripts/evaluate_nllfr.py',
    'research/fsm-author-nllfr-factorial-protocol.md',
    'scripts/audit_fsm_author_nllfr.py', 'scripts/audit_fsm_author_nllfr_budget.py'}
LEGAL_KEYS = ('starts', 'y_context', 'u_context', 'future_u')
STATE_KEYS = ('final_state', 'forecast_state', 'linear_seed_states', 'solved_context_start_states')


def relative(name):
    require(isinstance(name, str) and not Path(name).is_absolute() and '\\' not in name
        and all(p not in ('', '.', '..') for p in name.split('/')), 'canonical relative path')
    return ROOT/name


def read(path):
    return json.loads(Path(path).read_text())


def bound(value):
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
        and pin(value['path']) == value, 'descriptor drift')
    return Path(value['path'])


def inventory(folder):
    result = {}
    require(folder.is_dir() and not folder.is_symlink(), 'regular evidence directory')
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'symlink evidence')
        if path.is_file():
            result[str(path.relative_to(folder))] = {k: pin(path)[k] for k in ('bytes', 'sha256')}
    return result


def closed_audit(path, closure_path, source, process, registration, freeze=None):
    audit, closure = read(path), read(closure_path)
    require(audit['status'] == 'PASS' and audit['agreement'] is True, 'independent audit agreement required')
    require(closure['state'] == 'EXITED' and closure['observed_exit_code'] == 0
        and closure['success'] is True and closure['sources_unchanged'] is True
        and closure['inputs_unchanged'] is True and closure['error'] is None
        and closure['closure_error'] is None and closure['audit_output'] == pin(path)
        and closure['fit_process'] == pin(process) and closure['registration'] == pin(registration),
        'original audit closure required')
    require(closure['sources_before'] == closure['sources_after']
        and closure['inputs_before'] == closure['inputs_after'], 'audit before/after identities')
    require(closure['sources_before'][source] == pin(ROOT/source), 'held auditor identity')
    for item in (*closure['sources_before'].values(), *closure['inputs_before'].values(), closure['log']):
        bound(item)
    if freeze is not None:
        require(closure['freeze'] == pin(freeze), 'independent budget audit freeze')
    return audit


def qualified(cfg, paths):
    q = read(paths['producer_qualification'])
    definition = read(bound(q['definition']))
    require(q['status'] == 'PASS' and q['sources_before'] == q['sources_after'] == definition['sources']
        and set(definition['sources']) == set(cfg['source_sha256']), 'qualification source closure')
    for name, item in definition['sources'].items():
        require(bound(item) == relative(name) and item['sha256'] == cfg['source_sha256'][name],
            'qualified source identity')
        snapshot = pin(Path(q['definition']['path']).parent/'source'/name)
        require(all(snapshot[k] == item[k] for k in ('sha256', 'bytes')), 'qualification source snapshot')
    require([r['command'] for r in q['commands']] == definition['commands']
        and any('pytest' in r['command'] and 'research/fsm_author/tests/test_nllfr_factorial.py' in r['command']
            for r in q['commands']), 'qualification command scope')
    for row in q['commands']:
        require(row['returncode'] == 0, 'failed qualification')
        bound(row['log'])
    review = read(paths['source_review'])
    require(review['status'] == 'PASS' and review['reviewed_source_sha256'] == cfg['source_sha256']
        and review['qualification'] == cfg['prerequisites']['producer_qualification'], 'source review identity')


def parent_admission(paths):
    """Held admission functions hash opaque evidence, without numerical replay."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import audit_fsm_author_nllfr as old
    import audit_fsm_author_nllfr_budget as newer
    oa = closed_audit(paths['old_audit'], paths['old_audit_process'],
        'scripts/audit_fsm_author_nllfr.py', paths['old_process'], paths['old_registration'])
    old_plan, old_inputs, _, old_terminal, old_eval = old.authenticate(
        Path(oa['study']), paths['old_process'], paths['old_evaluation_process'])
    require(oa['inputs'] == old_inputs and old_terminal['status'] == old_eval['status'] == 'completed'
        and old_terminal['observed_exit_code'] == old_eval['observed_exit_code'] == 0,
        'old audited fit/evaluation admission')
    na = closed_audit(paths['new_audit'], paths['new_audit_process'],
        'scripts/audit_fsm_author_nllfr_budget.py', paths['new_process'], paths['new_registration'],
        paths['new_audit_freeze'])
    _, new_inputs, _, new_terminal, _, _ = newer.authenticate(
        Path(na['study']), paths['new_process'], paths['new_audit_freeze'])
    require(na['inputs'] == new_inputs and new_terminal['status'] == 'completed'
        and new_terminal['observed_exit_code'] == 0 and na['results']['budget_only_attributable'] is True
        and na['results']['prefix']['exact'] is True
        and na['results']['fit_status'] in ('complete', 'iteration_cap_reached')
        and na['scientific_status'] in ('FIT_ONLY_COMPLETE', 'FIT_ONLY_INCOMPLETE'),
        'finite exact-prefix fresh fit required before DEV')
    for label, audit, inputs in (('old', oa, old_inputs), ('new', na, new_inputs)):
        require(paths[label+'_final'] == Path(audit['study'])/'final.npz'
            and {k: pin(paths[label+'_final'])[k] for k in ('bytes', 'sha256')} == inputs['files']['final.npz'],
            'audited final role: '+label)
    require(paths['common_normalizer'] == ROOT/old_plan['prerequisites']['common_normalizer']['path']
        and sha(paths['common_normalizer']) == old_plan['prerequisites']['common_normalizer']['sha256'],
        'original common scoring normalizer')
    evaluation = Path(old_inputs['evaluation_study'])
    require(inventory(evaluation) == old_inputs['evaluation_files'], 'complete cached request inventory')
    for rid in DEV_IDS:
        for start in STARTS:
            prefix = f'evaluation/{rid}/{start:04d}'
            require(all(prefix+suffix in old_inputs['evaluation_files'] for suffix in ('.inputs.npz', '.npz', '.json')),
                'complete original legal request and target bank')
    banks = []
    reference_rows = read(old_inputs['reference_evaluations']['path'])
    require([(e['family'], e['seed']) for e in reference_rows] == old.ORDERED, 'fixed25 reference instances')
    reference_folder = Path(old_inputs['reference_evaluations']['path']).parent
    for entry in reference_rows:
        require([r['record_id'] for r in entry['rows']] == list(DEV_IDS)
            and all(r['status'] == 'complete' for r in entry['rows']), 'complete old reference roster')
        stem = entry['family']+(f'-{entry["seed"]}' if entry['seed'] is not None else '')
        for rid in DEV_IDS:
            banks.append({'family': entry['family'], 'seed': entry['seed'], 'record_id': rid,
                'file': pin(reference_folder/stem/'evaluation'/(rid+'.npz'))})
    bla_folder = (ROOT/old_plan['prerequisites']['bla_final_npz']['path']).parent
    banks.extend({'family': 'author_bla28', 'seed': None, 'record_id': rid,
        'file': pin(bla_folder/'evaluation'/(rid+'.npz'))} for rid in DEV_IDS)
    return {'old': old_inputs, 'new': new_inputs, 'old_audit': pin(paths['old_audit']),
        'new_audit': pin(paths['new_audit']), 'evaluation': str(evaluation),
        'new_fit_status': na['scientific_status'], 'new_fit_iterations': na['results']['iterations'],
        'prior_continuation': oa['results']['continuation'], 'reference_banks': banks}


def metadata_admission(registration, output=None):
    cfg = read(registration)
    require(set(cfg) == {'version', 'experiment', 'source_sha256', 'prerequisites', 'output', 'process_directory'}
        and cfg['version'] == VERSION and cfg['experiment'] == EXPERIMENT, 'fixed factorial recipe')
    require(MIN_SOURCES <= set(cfg['source_sha256']) and REQUIRED <= set(cfg['prerequisites']), 'source/input roster')
    for name, digest in cfg['source_sha256'].items():
        require(sha(relative(name)) == digest, 'source drift: '+name)
    paths = {}
    for name, item in cfg['prerequisites'].items():
        paths[name] = relative(item['path'])
        require(sha(paths[name]) == item['sha256'], 'prerequisite drift: '+name)
    if output is not None:
        require(output.resolve() == relative(cfg['output']).resolve(), 'registered output path')
    qualified(cfg, paths)
    admission = parent_admission(paths)
    return cfg, paths, admission


def load_arrays(path, names):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == set(names), 'saved array roster')
        return {key: archive[key].copy(order='K') for key in names}


def legal_cache(path, start):
    a = load_arrays(path, LEGAL_KEYS)
    require(a['starts'].dtype == np.int64 and a['starts'].shape == (1,)
        and int(a['starts'][0]) == start, 'request start identity')
    for key, shape in (('y_context', (1, 100, 3)), ('u_context', (1, 99, 3)), ('future_u', (1, 128, 3))):
        require(a[key].dtype == np.float64 and a[key].shape == shape and np.isfinite(a[key]).all(), 'legal input geometry')
        a[key].flags.writeable = False
    return a


def request(arrays, cached, iterations):
    """Time fresh owned legal copies and complete inference, never disk/targets."""
    start = time.perf_counter()
    y, u, future = (cached[k].copy(order='K') for k in ('y_context', 'u_context', 'future_u'))
    sliced = time.perf_counter()
    forecast, diagnostics = condition(arrays, y, u, iterations=iterations)
    conditioned = time.perf_counter()
    prediction, final = physical_rollout(arrays, future, forecast)
    ended = time.perf_counter()
    return {'prediction': prediction, 'final_state': final, 'forecast_state': forecast,
        'context': diagnostics, 'timing': {'request_ms': (ended-start)*1000, 'copy_ms': (sliced-start)*1000,
            'initializer_ms': (conditioned-sliced)*1000, 'rollout_ms': (ended-conditioned)*1000}}


def exact(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def json_tree(value):
    return json.loads(json.dumps(value, default=prior.json_value, allow_nan=False))


def prefix_check(short, long):
    if short is None or long is None:
        return {'status': 'unavailable', 'reason': 'numerical forecast failure'}
    a, b = short['context'], long['context']
    x, y = a['requests'][0], b['requests'][0]
    checks = {'same_linear_seed': exact(a['linear_seed_states'], b['linear_seed_states']),
        'same_direction_trial_prefix': exact(x['trace'], y['trace'][:len(x['trace'])]),
        'context_objective_nonincreasing': y['final_objective'] <= x['final_objective']}
    if x['status'] != 'ITERATION_CAP':
        checks['early_stop_identical'] = exact(x, y) and all(exact(short[k], long[k])
            for k in ('prediction', 'final_state', 'forecast_state')) and exact(
                a['solved_context_start_states'], b['solved_context_start_states'])
    return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'checks': checks,
        'short_status': x['status'], 'long_status': y['status']}


def old_replay_check(result, saved, row, standardized, cached):
    if result is None:
        return {'status': 'FAILED', 'reason': 'original complete request numerically failed'}
    checks = {'prediction': exact(standardized, saved['prediction']),
        'context': exact(json_tree(result['context']), row['context']),
        'starts': exact(cached['starts'], saved['starts'])}
    for key in STATE_KEYS:
        value = result[key] if key in result else result['context'][key]
        checks[key] = exact(value, saved[key])
    for key in ('y_context', 'u_context', 'future_u'):
        checks[key] = exact(cached[key], saved[key])
    return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'checks': checks}


def standardized_score(prediction, target, scale):
    require(prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[-1] == 3
        and min(prediction.shape[:2]) > 0 and prediction.dtype == target.dtype == np.float64,
        'standardized score geometry')
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'nonfinite score inputs')
    require(isinstance(scale, np.ndarray) and scale.dtype == np.float64 and scale.shape == (3,)
        and np.isfinite(scale).all() and (scale > 0).all(), 'positive scoring scales')
    with np.errstate(over='ignore', invalid='ignore'):
        square = (prediction-target)**2
        mse = float(square.mean())
        channel = np.sqrt(square.mean(axis=(0, 1)))
        native = channel*scale
    require(math.isfinite(mse) and np.isfinite(channel).all() and np.isfinite(native).all(), 'score overflow')
    return {'rmse': math.sqrt(mse), 'mse': mse, 'per_channel_rmse': channel.tolist(),
        'native_output_per_channel_rmse': native.tolist(), 'requests': len(prediction), 'horizon': prediction.shape[1]}


def error_record(exc):
    return {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}


def timing_schedule():
    rows = []
    for slot, (rid, start) in enumerate((r, s) for r in DEV_IDS for s in (0, 7936)):
        order = CELLS[slot % 4:]+CELLS[:slot % 4]
        rows.extend({'slot': slot, 'position': pos, 'cell': cell, 'record_id': rid, 'start': start}
            for pos, cell in enumerate(order))
    return rows


def timed_attempt(identity, arrays, cached):
    try:
        result = request(arrays, cached, int(identity['cell'][-2:]))
        return {**identity, 'status': 'complete', **result['timing'],
            'context': result['context'], 'error': None}
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        if not prior.numerical_failure(exc):
            raise
        return {**identity, 'status': 'failed', 'error': error_record(exc)}


def contrasts(cells):
    means = {c: cells[c]['mean_rmse'] for c in CELLS}
    pairs = {'training_at16': ('old16', 'new16'), 'training_at64': ('old64', 'new64'),
             'inference_old': ('old16', 'old64'), 'inference_new': ('new16', 'new64')}
    result = {'cell_means': means, 'sign': 'positive reduction means lower forecast error', 'effects': {}}
    for name, (a, b) in pairs.items():
        delta = means[a]-means[b] if means[a] is not None and means[b] is not None else None
        result['effects'][name] = {'from': a, 'to': b, 'denominator': means[a], 'absolute_reduction': delta,
            'relative_reduction_percent': 100*delta/means[a] if delta is not None and means[a] > 0 else None}
    e = result['effects']
    values = [e[k]['absolute_reduction'] for k in ('training_at64', 'training_at16')]
    result['interaction_absolute'] = values[0]-values[1] if all(v is not None for v in values) else None
    result['interaction_definition'] = '(old64-new64)-(old16-new16), absolute only'
    return result


def continuation(previous, cells, new_fit_complete, parity_ok):
    from copy import deepcopy
    result = deepcopy(previous)
    means, records, seeds = result['family_means'], result['per_record_means'], result['per_seed_means']
    eligible = new_fit_complete and parity_ok and all(cells[c]['mean_rmse'] is not None
        and cells[c]['median_request_ms'] is not None for c in CELLS)
    for cell in ('new16', 'new64'):
        if new_fit_complete and parity_ok and cells[cell]['mean_rmse'] is not None:
            means[cell] = cells[cell]['mean_rmse']
            records[cell] = {r['record_id']: r['rmse'] for r in cells[cell]['rows']}
            seeds[cell] = {'None': means[cell]}
    candidate = result['candidate']
    strongest = min((f for f in means if f != candidate), key=lambda f: (means[f], f), default=None)
    gates = dict.fromkeys(previous['conditions'], False)
    if eligible and strongest is not None:
        c, b = records[candidate], records[strongest]
        gates['five_percent_below_strongest_control'] = means[candidate] <= .95*means[strongest]
        gates['every_seed_below_strongest_control'] = all(value < seeds[strongest].get(seed, seeds[strongest].get('None'))
            for seed, value in seeds[candidate].items())
        gates['no_record_over_two_percent_strongest_control'] = all(c[r] <= 1.02*b[r] for r in DEV_IDS)
        gates['both_amplitudes_below_strongest_control'] = all(np.mean([c[r] for r in DEV_IDS if r.startswith(a)])
            < np.mean([b[r] for r in DEV_IDS if r.startswith(a)]) for a in ('100mV', '200mV'))
    result.update(conditions=gates, passed=sum(gates.values()), total=4, reference_complete=eligible,
        status='CONTINUE_REFERENCE_CHECK' if all(gates.values()) else 'DO_NOT_CONTINUE_REFERENCE_CHECK',
        strongest_control=strongest, blocking_scope='New fit completion, all four forecast/cost cells and implementation parity required')
    result['per_amplitude_means'] = {f: {a: float(np.mean([v for r, v in rec.items() if r.startswith(a)]))
        for a in ('100mV', '200mV')} for f, rec in records.items()}
    return result


def rescore_references(admission, norm, request_targets):
    """Rescore unchanged normalized banks, never invoke a reference model."""
    from copy import deepcopy
    previous = deepcopy(admission['prior_continuation'])
    groups, rows = {}, []
    for item in admission['reference_banks']:
        with np.load(item['file']['path'], allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in ('prediction', 'target', 'starts')}
        require(values['starts'].dtype == np.int64 and np.array_equal(values['starts'], np.array(STARTS)), 'reference start alignment')
        rid = item['record_id']
        require(values['prediction'].shape == values['target'].shape == (32, 128, 3)
            and exact(values['target'], request_targets[rid]), 'all reference targets match cached targets')
        row = {**{k: item[k] for k in ('family', 'seed', 'record_id')}, 'status': 'complete',
            **standardized_score(values['prediction'], values['target'], norm['y_scale'])}
        rows.append(row)
        groups.setdefault(item['family'], {}).setdefault(item['seed'], []).append(row)
    for family, seeds in groups.items():
        require(all([r['record_id'] for r in own] == list(DEV_IDS) for own in seeds.values()), 'reference group geometry')
        record = {rid: float(np.mean([own[i]['rmse'] for own in seeds.values()])) for i, rid in enumerate(DEV_IDS)}
        previous['family_means'][family] = float(np.mean(list(record.values())))
        previous['per_record_means'][family] = record
        previous['per_seed_means'][family] = {str(s): float(np.mean([r['rmse'] for r in own])) for s, own in seeds.items()}
    previous['per_amplitude_means'] = {f: {a: float(np.mean([v for r, v in rec.items() if r.startswith(a)]))
        for a in ('100mV', '200mV')} for f, rec in previous['per_record_means'].items()}
    return {'banks': admission['reference_banks'], 'rows': rows, 'rescored_continuation': previous,
        'historical_continuation': admission['prior_continuation'],
        'scope': '312 unchanged normalized prediction/target banks rescored; no model calls or historical report edits'}


def evaluate(models, cache, norm, reference_root, output, admission):
    """All fixed slots attempted. Failed forecasts never receive a survivor mean."""
    cells = {c: {'rows': [], 'requests': [], 'context_status_counts': {}} for c in CELLS}
    parity = {'old16': [], 'within_checkpoint': []}
    request_targets = {}
    before = {k: {n: v.copy(order='K') for n, v in a.items()} for k, a in models.items()}
    for rid in DEV_IDS:
        per_cell = {c: [] for c in CELLS}
        for start in STARTS:
            outcomes = {}
            for cell in CELLS:
                folder = output/'cells'/cell/'evaluation'/rid
                folder.mkdir(parents=True, exist_ok=True)
                label = f'{start:04d}'
                row = {'cell': cell, 'record_id': rid, 'start': start}
                result = None
                try:
                    result = request(models[cell[:3]], cache[rid, start], int(cell[-2:]))
                    # Only after inference returns are future targets/reference outputs decoded.
                    with np.load(reference_root/'evaluation'/rid/(label+'.npz'), allow_pickle=False) as archive:
                        target = archive['target'].copy()
                        saved = {key: archive[key].copy(order='K') for key in
                            ('prediction', 'starts', *STATE_KEYS, 'y_context', 'u_context', 'future_u')} if cell == 'old16' else None
                    require(target.dtype == np.float64 and target.shape == (1, 128, 3) and np.isfinite(target).all(), 'cached target geometry')
                    previous_target = request_targets.setdefault((rid, start), target)
                    require(exact(previous_target, target), 'target changed across cells')
                    standardized = (result['prediction']-norm['y_mean'])/norm['y_scale']
                    payload = {'prediction': standardized, 'physical_prediction': result['prediction'], 'target': target,
                        'starts': np.array([start], dtype=np.int64), **{k: result[k] for k in STATE_KEYS[:2]},
                        **{k: result['context'][k] for k in STATE_KEYS[2:]}}
                    # Raw returned forecast/state evidence precedes score/parity validation.
                    np.savez_compressed(folder/(label+'.npz'), **payload)
                    row.update(context=result['context'], diagnostic_request_timing=result['timing'])
                    row.update(status='complete', **standardized_score(standardized, target, norm['y_scale']),
                        )
                    per_cell[cell].append(payload)
                    if cell == 'old16':
                        check = old_replay_check(result, saved, read(reference_root/'evaluation'/rid/(label+'.json')),
                            standardized, cache[rid, start])
                        parity['old16'].append({'record_id': rid, 'start': start, **check})
                except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                    if not prior.numerical_failure(exc):
                        raise
                    row.update(status='failed', error=error_record(exc))
                    result = None
                    if cell == 'old16':
                        parity['old16'].append({'record_id': rid, 'start': start,
                            'status': 'FAILED', 'reason': 'original complete request numerically failed'})
                write(folder/(label+'.json'), row)
                cells[cell]['requests'].append({k: row[k] for k in ('cell', 'record_id', 'start', 'status')})
                if result is not None:
                    status = result['context']['requests'][0]['status']
                    counts = cells[cell]['context_status_counts']
                    counts[status] = counts.get(status, 0)+1
                outcomes[cell] = result
            for checkpoint in ('old', 'new'):
                parity['within_checkpoint'].append({'checkpoint': checkpoint, 'record_id': rid, 'start': start,
                    **prefix_check(outcomes[checkpoint+'16'], outcomes[checkpoint+'64'])})
        for cell in CELLS:
            parts = per_cell[cell]
            complete = len(parts) == 32
            row = {'record_id': rid, 'status': 'complete' if complete else 'incomplete',
                'completed_requests': len(parts), 'expected_requests': 32}
            if complete:
                merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
                np.savez_compressed(output/'cells'/cell/'evaluation'/(rid+'.npz'), **merged)
                try:
                    row.update(standardized_score(merged['prediction'], merged['target'], norm['y_scale']))
                except ValueError as exc:
                    if not prior.numerical_failure(exc):
                        raise
                    row.update(status='incomplete', error=error_record(exc))
            cells[cell]['rows'].append(row)
        require(all(exact(models[k], v) for k, v in before.items()), 'model mutation')
        event(output, 'record_closed', record_id=rid, cells={c: cells[c]['rows'][-1]['status'] for c in CELLS})
    timing = {'warmups': [], 'samples': [], 'scope': 'Fresh legal copies, normalization, seed, full GN/SVD and physical H128 forecast; disk/scoring excluded'}
    for cell in CELLS:
        identity = {'cell': cell, 'record_id': DEV_IDS[0], 'start': 0}
        timing['warmups'].append(timed_attempt(identity, models[cell[:3]], cache[DEV_IDS[0], 0]))
    for identity in timing_schedule():
        timing['samples'].append(timed_attempt(identity, models[identity['cell'][:3]], cache[identity['record_id'], identity['start']]))
    require(all(exact(models[k], v) for k, v in before.items()), 'timing mutated model')
    for cell in CELLS:
        own = cells[cell]
        own['mean_rmse'] = float(np.mean([r['rmse'] for r in own['rows']])) if all(r['status'] == 'complete' for r in own['rows']) else None
        own['amplitude_mean_rmse'] = {}
        for amplitude in ('100mV', '200mV'):
            rows = [r for r in own['rows'] if r['record_id'].startswith(amplitude)]
            own['amplitude_mean_rmse'][amplitude] = float(np.mean([r['rmse'] for r in rows])) if len(rows) == 6 and all(r['status'] == 'complete' for r in rows) else None
        samples = [r for r in timing['samples'] if r['cell'] == cell]
        warm = [r for r in timing['warmups'] if r['cell'] == cell]
        own['median_request_ms'] = float(np.median([r['request_ms'] for r in samples])) if len(samples) == 24 and all(r['status'] == 'complete' for r in samples+warm) else None
        a = models[cell[:3]]
        own['model_numeric_bytes'] = sum(v.nbytes for v in a.values())
        own['state_bytes'], own['policy_bytes'] = len(a['A'])*8, 9*8
        own['persistent_numeric_bytes'] = own['model_numeric_bytes']+own['state_bytes']+own['policy_bytes']
        own['failure_counts'] = {'forecast': sum(r['status'] != 'complete' for r in own['requests']),
            'record': sum(r['status'] != 'complete' for r in own['rows']),
            'warmup': sum(r['status'] != 'complete' for r in warm),
            'timing': sum(r['status'] != 'complete' for r in samples)}
        own['storage_scope'] = 'All19 numeric model arrays, retained state and nine C/H/solver-policy scalars; no prepared cache'
    parity_ok = all(r['status'] == 'PASS' for rows in parity.values() for r in rows)
    # If all four calls at a slot failed, its target was never needed. Load only
    # after all forecast attempts, for the separately declared reference scoring.
    for rid in DEV_IDS:
        for start in STARTS:
            if (rid, start) not in request_targets:
                with np.load(reference_root/'evaluation'/rid/f'{start:04d}.npz', allow_pickle=False) as archive:
                    request_targets[rid, start] = archive['target'].copy()
    targets = {rid: np.concatenate([request_targets[rid, s] for s in STARTS]) for rid in DEV_IDS}
    refs = rescore_references(admission, norm, targets)
    write(output/'references.json', refs)
    result = {'cells': cells, 'timing': timing, 'parity': parity, 'parity_pass': parity_ok,
        'contrasts': contrasts(cells), 'prior_continuation': admission['prior_continuation'],
        'continuation': continuation(refs['rescored_continuation'], cells,
            admission['new_fit_status'] == 'FIT_ONLY_COMPLETE', parity_ok)}
    write(output/'evaluation.json', result)
    return result


def run(registration, output):
    cfg, paths, admission = metadata_admission(registration, output)
    before = prior.identities(cfg)
    write(output/'identity-before.json', before)
    write(output/'admission.json', admission)
    for name in cfg['source_sha256']:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative(name).read_bytes())
    reference = Path(admission['evaluation'])
    allowed = {paths[k].resolve() for k in ('old_final', 'new_final', 'common_normalizer')}
    allowed.update((reference/'evaluation'/rid/(f'{start:04d}'+suffix)).resolve()
        for rid in DEV_IDS for start in STARTS for suffix in ('.inputs.npz', '.npz'))
    allowed.update(Path(row['file']['path']).resolve() for row in admission['reference_banks'])
    def guard(name, args):
        if name == 'socket.connect':
            raise RuntimeError('network forbidden during factorial evaluation')
        if name == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.suffix.lower() in ('.npz', '.npy', '.mat', '.zip'):
                require(p in allowed or p.is_relative_to(output), 'unregistered numerical file access')
    # Metadata hashing runs before this numerical access guard and once after it is disabled.
    active = [True]
    def hook(name, args):
        if active[0]:
            guard(name, args)
    sys.addaudithook(hook)
    models = {k: load_arrays(paths[k+'_final'], NAMES) for k in ('old', 'new')}
    require(all(validate(a) == (28, 16, 8, 64) for a in models.values()), 'full final architecture')
    norm = load_arrays(paths['common_normalizer'], ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    require(all(v.shape == (3,) and v.dtype == np.float64 and np.isfinite(v).all() for v in norm.values())
        and (norm['y_scale'] > 0).all(), 'common normalizer')
    for key, name in (('old_final', 'old-final.npz'), ('new_final', 'new-final.npz'), ('common_normalizer', 'common-normalizer.npz')):
        (output/name).write_bytes(paths[key].read_bytes())
    cache = {}
    for rid in DEV_IDS:
        for start in STARTS:
            source = reference/'evaluation'/rid/f'{start:04d}.inputs.npz'
            cache[rid, start] = legal_cache(source, start)
            dest = output/'requests'/rid/source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
    result = evaluate(models, cache, norm, reference, output, admission)
    summary = {'status': 'FACTORIAL_COMPLETE' if result['parity_pass'] and all(
        v['mean_rmse'] is not None and v['median_request_ms'] is not None for v in result['cells'].values()) else 'FACTORIAL_INCOMPLETE',
        'new_fit_status': admission['new_fit_status'], 'new_fit_iterations': admission['new_fit_iterations'],
        'cells': {c: {k: result['cells'][c][k] for k in ('mean_rmse', 'amplitude_mean_rmse', 'median_request_ms', 'persistent_numeric_bytes', 'context_status_counts', 'failure_counts')} for c in CELLS},
        'counts': {'forecast_attempts': sum(len(v['requests']) for v in result['cells'].values()),
            'record_slots': sum(len(v['rows']) for v in result['cells'].values()),
            'warmups': len(result['timing']['warmups']), 'timed_requests': len(result['timing']['samples'])},
        'continuation': result['continuation'], 'contrasts': result['contrasts'],
        'scope': 'Exposed cached DEV; no fitting, selection, raw/archive decode or control claim. Old cells remain incomplete diagnostics.'}
    write(output/'summary.json', summary)
    active[0] = False
    after_cfg, _, after_admission = metadata_admission(registration, output)
    require(after_cfg == cfg and after_admission == admission, 'source/evidence drift during evaluation')
    after = prior.identities(cfg)
    require(after == before, 'installed runtime drift')
    write(output/'identity-after.json', after)
    event(output, 'evaluation_closed', **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'run from OpenJev root')
    # Direct invocation also rejects an unregistered destination before writes;
    # the supervising original process retains admission failures in its log.
    metadata_admission(args.registration, args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'started.json', {'registration_sha256': sha(args.registration), 'pid': os.getpid()})
    try:
        run(args.registration, args.output.resolve())
    except Exception as exc:
        write(args.output/'failure.json', error_record(exc))
        raise


if __name__ == '__main__':
    main()
