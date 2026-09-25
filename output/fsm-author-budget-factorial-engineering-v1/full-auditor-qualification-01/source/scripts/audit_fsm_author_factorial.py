# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent four-cell NL-LFR replay; no production numerical imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = 'scripts/fsm_nllfr_budget_audit_math.py'
MATH_SHA = 'd4454df6f8854d1895d7edfee75dd1e36a58ec616fb12537a7424359a538f8c6'
HELPER = 'scripts/audit_fsm_author_nllfr.py'
HELPER_SHA = '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32'
for _name, _sha in ((MATH, MATH_SHA), (HELPER, HELPER_SHA)):
    if hashlib.sha256((ROOT/_name).read_bytes()).hexdigest() != _sha:
        raise ValueError('held independent numerical source changed')
_spec = importlib.util.spec_from_file_location('factorial_independent_math', ROOT/MATH)
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
old = replay.old
require, read, pin, descriptor, close = old.require, old.read, old.pin, old.descriptor, old.close
finite, load_arrays = old.finite, old.load_arrays
CELLS = ('old16', 'old64', 'new16', 'new64')
DEV, STARTS = old.DEV, tuple(range(0, 7937, 256))
SOURCES = ('scripts/audit_fsm_author_factorial.py', 'tests/test_audit_fsm_author_factorial.py')
VERSION = 'fsm-author-nllfr-factorial-v1'
TOLERANCES = old.TOLERANCES
INPUT_KEYS = ('starts', 'y_context', 'u_context', 'future_u')
BANK_KEYS = ('prediction', 'physical_prediction', 'target', 'starts', 'final_state',
             'forecast_state', 'linear_seed_states', 'solved_context_start_states')


def exact(actual, expected, label='exact evidence'):
    """Array equality includes dtype and signed-zero bits; metadata is recursive."""
    if isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(), label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+' schema')
        for key in expected:
            exact(actual[key], expected[key], label+'/'+key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+' length')
        for a, e in zip(actual, expected, strict=True):
            exact(a, e, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label)


def aggregate(rows):
    require([r['record_id'] for r in rows] == list(DEV), 'all twelve ordered records')
    valid = all(r['status'] == 'complete' and type(r.get('rmse')) in (float, int)
                and math.isfinite(r['rmse']) and r['rmse'] >= 0 for r in rows)
    if not valid:
        return None
    return float(np.mean([r['rmse'] for r in rows]))


def compare_budgets(short, long, short_bank, long_bank):
    """Compare saved production paths exactly; independent replay is separate."""
    require(short['policy'] == replay.policy(iterations=16)
            and long['policy'] == replay.policy(iterations=64), 'two fixed policies')
    exact(short['linear_seed_states'], long['linear_seed_states'], 'same per-checkpoint seed')
    exact(short['linear_seed_diagnostics'], long['linear_seed_diagnostics'], 'same seed derivation')
    a, b = short['requests'][0], long['requests'][0]
    require(len(short['requests']) == len(long['requests']) == 1, 'single request diagnostic')
    require(len(b['trace']) >= len(a['trace']), 'long solve preserves available prefix')
    exact(a['trace'], b['trace'][:len(a['trace'])], 'direction/trial prefix')
    require(b['final_objective'] <= a['final_objective'], 'longer context objective worsened')
    if a['status'] != 'ITERATION_CAP':
        exact(a, b, 'identical early stop')
        for key in ('prediction', 'final_state', 'forecast_state', 'solved_context_start_states'):
            exact(short_bank[key], long_bank[key], 'same early-stop '+key)
    else:
        require(a['directions_considered'] == 16, 'sixteen-direction cap')
    return True


def bound(value):
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
            and descriptor(value['path']) == value, 'bound evidence descriptor')
    return Path(value['path'])


def source_freeze(freeze):
    held = read(freeze)
    require(held['status'] == 'FROZEN_BEFORE_EMPIRICAL_AUDIT'
            and set(held['sources']) == {*SOURCES, MATH, HELPER}, 'independent source freeze')
    for name, value in held['sources'].items():
        require(bound(value) == ROOT/name, 'current independent source')
    require(held['sources'][MATH]['sha256'] == MATH_SHA
            and held['sources'][HELPER]['sha256'] == HELPER_SHA, 'held independent equations')
    expected = {name: held['sources'][name] for name in SOURCES}
    q = read(bound(held['qualification']))
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True
            and q['sources_before'] == q['sources_after'] == expected, 'original fabricated qualification')
    commands = [['.venv/bin/ruff', 'check', *SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', SOURCES[1]]]
    require(len(q['commands']) == 2, 'two original qualification commands')
    for row, command in zip(q['commands'], commands, strict=True):
        require(row['command'] == command and row['returncode'] == 0, 'qualified command')
        bound(row['log'])
    pre = read(bound(q['preflight']))
    require(pre['sources'] == expected and pre['commands'] == commands
            and set(pre['snapshots']) == set(SOURCES), 'qualification preflight')
    for name, value in pre['snapshots'].items():
        bound(value)
        require(value['sha256'] == expected[name]['sha256'], 'qualified original snapshot')
    return held, bound(held['registration'])


def reference_groups(evaluations, bla_rows):
    require([(r['family'], r['seed']) for r in evaluations] == old.ORDERED, '25 fixed reference slots')
    groups = {f: [r for r in evaluations if r['family'] == f] for f in old.FAMILIES}
    groups[old.BLA] = [{'seed': None, 'rows': bla_rows}]
    return groups


def decisions(evaluations, bla_rows, cells, *, fit_complete, matrix_complete, parity_pass=True):
    """Quality eligibility is independent of timing; missing cost blocks promotion."""
    require(type(fit_complete) is type(matrix_complete) is bool and set(cells) == set(CELLS),
            'explicit completion and four cells')
    groups = reference_groups(evaluations, bla_rows)
    groups.update({name: [{'seed': None, 'rows': cells[name]['rows']}]
                   for name in ('new16', 'new64') if fit_complete and parity_pass})
    means, records, seeds = {}, {}, {}
    for name, members in groups.items():
        if not members or any(aggregate(item['rows']) is None for item in members):
            continue
        records[name] = {rid: float(np.mean([item['rows'][j]['rmse'] for item in members]))
                         for j, rid in enumerate(DEV)}
        seeds[name] = {str(item['seed']): aggregate(item['rows']) for item in members}
        means[name] = float(np.mean(list(records[name].values())))
    eligible = [name for name in means if name != old.CANDIDATE]
    strongest = min(eligible, key=lambda name: (means[name], name), default=None)
    gates = dict.fromkeys(('five_percent_below_strongest_control', 'every_seed_below_strongest_control',
                          'no_record_over_two_percent_strongest_control',
                          'both_amplitudes_below_strongest_control'), False)
    complete = (fit_complete and matrix_complete and set(means) == {*old.FAMILIES, old.BLA, 'new16', 'new64'})
    if complete and strongest is not None:
        candidate, baseline = records[old.CANDIDATE], records[strongest]
        gates['five_percent_below_strongest_control'] = means[old.CANDIDATE] <= .95*means[strongest]
        gates['every_seed_below_strongest_control'] = all(
            seeds[old.CANDIDATE][str(s)] < seeds[strongest].get(str(s), seeds[strongest].get('None'))
            for s in old.SEEDS)
        gates['no_record_over_two_percent_strongest_control'] = all(candidate[r] <= 1.02*baseline[r] for r in DEV)
        gates['both_amplitudes_below_strongest_control'] = all(
            np.mean([candidate[r] for r in DEV if r.startswith(a)])
            < np.mean([baseline[r] for r in DEV if r.startswith(a)]) for a in ('100mV', '200mV'))
    return {'status': 'CONTINUE_REFERENCE_CHECK' if all(gates.values()) else 'DO_NOT_CONTINUE_REFERENCE_CHECK',
            'reference_complete': complete, 'candidate': old.CANDIDATE, 'strongest_control': strongest,
            'conditions': gates, 'passed': sum(gates.values()), 'total': 4,
            'family_means': means, 'per_record_means': records, 'per_seed_means': seeds,
            'per_amplitude_means': {name: {a: float(np.mean([v for r, v in rows.items() if r.startswith(a)]))
                for a in ('100mV', '200mV')} for name, rows in records.items()},
            'seed_rule': 'Paired seed for stochastic controls; common deterministic score otherwise. No new candidate selection.',
            'blocking_scope': 'New fit completion, all four forecast/cost cells and implementation parity required'}


def contrasts(means):
    require(set(means) == set(CELLS), 'all four means')
    for value in means.values():
        require(value is None or (type(value) in (float, int) and math.isfinite(value) and value >= 0),
                'finite nonnegative mean or explicit missing')
    effects = {}
    for name, a, b in (('training_at16', 'old16', 'new16'), ('training_at64', 'old64', 'new64'),
                        ('inference_old', 'old16', 'old64'), ('inference_new', 'new16', 'new64')):
        before, after = means[a], means[b]
        delta = None if before is None or after is None else before-after
        effects[name] = {'from': a, 'to': b, 'absolute_reduction': delta,
                         'denominator': before,
                         'relative_reduction_percent': 100*delta/before if delta is not None and before > 0 else None}
    interaction = (effects['training_at64']['absolute_reduction']-effects['training_at16']['absolute_reduction']
                   if all(v is not None for v in means.values()) else None)
    return {'cell_means': means, 'sign': 'positive reduction means lower forecast error',
            'effects': effects, 'interaction_absolute': interaction,
            'interaction_definition': '(old64-new64)-(old16-new16), absolute only'}


def timing_summary(timing):
    require(set(timing) == {'warmups', 'samples', 'scope'} and timing['scope'] ==
            'Fresh legal copies, normalization, seed, full GN/SVD and physical H128 forecast; disk/scoring excluded', 'timing schema')
    warmups, samples = timing['warmups'], timing['samples']
    require([(r['cell'], r['record_id'], r['start']) for r in warmups]
            == [(cell, DEV[0], 0) for cell in CELLS], 'four first-request warmup attempts')
    expected = []
    for slot, (rid, start) in enumerate((r, s) for r in DEV for s in (0, 7936)):
        rotation = CELLS[slot % 4:]+CELLS[:slot % 4]
        expected.extend((slot, position, cell, rid, start) for position, cell in enumerate(rotation))
    require([(r['slot'], r['position'], r['cell'], r['record_id'], r['start']) for r in samples] == expected,
            '96 exact rotated timing slots')
    for row in (*warmups, *samples):
        require(row['status'] in ('complete', 'failed'), 'explicit timing outcome')
        if row['status'] == 'complete':
            validate_timing(row)
        else:
            error_record(row['error'])
            require(not any(k in row for k in ('request_ms', 'copy_ms', 'initializer_ms', 'rollout_ms')),
                    'no invented failed timing values')
    medians = {}
    for cell in CELLS:
        selected = [r for r in samples if r['cell'] == cell]
        complete = (next(r for r in warmups if r['cell'] == cell)['status'] == 'complete'
                    and all(r['status'] == 'complete' for r in selected))
        medians[cell] = float(np.median([r['request_ms'] for r in selected])) if complete else None
    return medians, all(value is not None for value in medians.values())


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
    'research/fsm_author/scripts/evaluate_nllfr.py', HELPER,
    'scripts/audit_fsm_author_nllfr_budget.py'}


def closed_audit(path, closure_path, source, process, registration, freeze=None):
    evidence, terminal = read(path), read(closure_path)
    require(evidence['status'] == 'PASS' and evidence['agreement'] is True, 'parent audit agreement')
    require(terminal['state'] == 'EXITED' and terminal['observed_exit_code'] == 0
            and terminal['success'] is True and terminal['sources_unchanged'] is True
            and terminal['inputs_unchanged'] is True and terminal['error'] is None
            and terminal['closure_error'] is None, 'original successful audit terminal')
    require(terminal['audit_output'] == descriptor(path) and terminal['fit_process'] == descriptor(process)
            and terminal['registration'] == descriptor(registration), 'original audit input/output joins')
    require(terminal['sources_before'] == terminal['sources_after']
            and terminal['inputs_before'] == terminal['inputs_after'], 'original audit closure')
    require(terminal['sources_before'][source] == descriptor(ROOT/source), 'original audit source identity')
    for value in (*terminal['sources_before'].values(), *terminal['inputs_before'].values(), terminal['log']):
        bound(value)
    if freeze is not None:
        require(terminal['freeze'] == descriptor(freeze), 'parent independent source freeze')
    return evidence


def parent_admission(paths):
    oa = closed_audit(paths['old_audit'], paths['old_audit_process'], HELPER,
                      paths['old_process'], paths['old_registration'])
    _plan, inputs, prior_paths, fit_terminal, eval_terminal = old.authenticate(
        Path(oa['study']), paths['old_process'], paths['old_evaluation_process'])
    require(oa['inputs'] == inputs and fit_terminal['status'] == eval_terminal['status'] == 'completed',
            'original audited old fit and evaluation')
    require(oa['results']['fit_status'] == 'iteration_cap_reached', 'old checkpoint remains capped')
    na = closed_audit(paths['new_audit'], paths['new_audit_process'],
        'scripts/audit_fsm_author_nllfr_budget.py', paths['new_process'],
        paths['new_registration'], paths['new_audit_freeze'])
    ni = na['inputs']
    require(ni['registration'] == descriptor(paths['new_registration'])
            and ni['process'] == descriptor(paths['new_process'])
            and ni['freeze'] == descriptor(paths['new_audit_freeze']), 'new audit original identities')
    np_ = read(paths['new_process'])
    require(np_['status'] == 'completed' and np_['observed_exit_code'] == 0
            and np_['end_identity_matches'] is True
            and np_['artifacts'] == ni['files'] == old.inventory(Path(na['study'])), 'new full closed fit inventory')
    require(na['results']['budget_only_attributable'] is True and na['results']['prefix']['exact'] is True
            and na['results']['prefix']['required'] == na['results']['prefix']['compared'] == 10000
            and na['results']['fit_status'] in ('complete', 'iteration_cap_reached'),
            'exact attributed budget fit before DEV')
    require(na['scientific_status'] == ('FIT_ONLY_COMPLETE' if na['results']['fit_status'] == 'complete'
                                     else 'FIT_ONLY_INCOMPLETE'), 'new completion distinct from attribution')
    require(ni['prerequisites']['parent_audit'] == descriptor(paths['old_audit'])
            and ni['prerequisites']['parent_process'] == descriptor(paths['old_process'])
            and ni['prerequisites']['parent_evaluation_process'] == descriptor(paths['old_evaluation_process']),
            'old/new parent lineage')
    for label, evidence in (('old', oa), ('new', na)):
        path = paths[label+'_final']
        require(path == Path(evidence['study'])/'final.npz'
                and pin(path) == evidence['inputs']['files']['final.npz'], 'correct final checkpoint role')
    require(paths['common_normalizer'] == prior_paths['common_normalizer'], 'original common normalizer role')
    evaluation = Path(inputs['evaluation_study'])
    require(old.inventory(evaluation) == inputs['evaluation_files'], 'all audited cached request bytes')
    for rid in DEV:
        for start in STARTS:
            stem = f'evaluation/{rid}/{start:04d}'
            require(all(stem+s in inputs['evaluation_files'] for s in ('.json', '.npz', '.inputs.npz')),
                    'complete original request cache')
    banks = []
    reference_rows = read(prior_paths['parent_evaluations'])
    require([(e['family'], e['seed']) for e in reference_rows] == old.ORDERED, 'fixed inherited25 instances')
    for entry in reference_rows:
        require([r['record_id'] for r in entry['rows']] == list(DEV)
                and all(r['status'] == 'complete' for r in entry['rows']), 'complete inherited reference roster')
        stem = entry['family']+(f'-{entry["seed"]}' if entry['seed'] is not None else '')
        for rid in DEV:
            banks.append({'family': entry['family'], 'seed': entry['seed'], 'record_id': rid,
                'file': descriptor(prior_paths['parent_evaluations'].parent/stem/'evaluation'/(rid+'.npz'))})
    banks.extend({'family': old.BLA, 'seed': None, 'record_id': rid,
        'file': descriptor(prior_paths['bla_final_npz'].parent/'evaluation'/(rid+'.npz'))} for rid in DEV)
    return {'old': inputs, 'new': ni, 'old_audit': descriptor(paths['old_audit']),
            'new_audit': descriptor(paths['new_audit']), 'evaluation': str(evaluation),
            'new_fit_status': na['scientific_status'], 'new_fit_iterations': na['results']['iterations'],
            'prior_continuation': oa['results']['continuation'], 'reference_banks': banks}, prior_paths


def qualification(plan, paths):
    q = read(paths['producer_qualification'])
    definition = read(bound(q['definition']))
    require(q['status'] == 'PASS' and q['sources_before'] == q['sources_after'] == definition['sources']
            and set(definition['sources']) == set(plan['source_sha256']), 'producer source qualification')
    for name, value in definition['sources'].items():
        require(bound(value) == ROOT/old.relative(name) and value['sha256'] == plan['source_sha256'][name],
                'producer qualified source')
        require(pin(Path(q['definition']['path']).parent/'source'/name)
                == {k: value[k] for k in ('sha256', 'bytes')}, 'producer source snapshot')
    require([row['command'] for row in q['commands']] == definition['commands']
            and any('pytest' in row['command'] and 'research/fsm_author/tests/test_nllfr_factorial.py' in row['command']
                    for row in q['commands']), 'producer qualified commands')
    for row in q['commands']:
        require(row['returncode'] == 0, 'successful producer qualification')
        bound(row['log'])
    peer = read(paths['source_review'])
    require(peer['status'] == 'PASS' and peer['reviewed_source_sha256'] == plan['source_sha256']
            and peer['qualification'] == plan['prerequisites']['producer_qualification'], 'producer peer review')


def reference_scores(paths, common, parent_targets):
    """Same saved predictions, newly derived standardized-difference metrics."""
    original = read(paths['parent_evaluations'])
    require([(e['family'], e['seed']) for e in original] == old.ORDERED, 'frozen25 reference slots')
    rebuilt, count = [], 0
    for entry in original:
        stem = entry['family']+(f'-{entry["seed"]}' if entry['seed'] is not None else '')
        require([r['record_id'] for r in entry['rows']] == list(DEV), 'reference record roster')
        rows = []
        for row in entry['rows']:
            if row['status'] != 'complete':
                require(row['status'] in ('failed', 'not_run'), 'explicit inherited failure')
                rows.append(row); continue
            bank = load_arrays(paths['parent_evaluations'].parent/stem/'evaluation'/(row['record_id']+'.npz'),
                               ('prediction', 'target', 'starts'))
            exact(bank['starts'], np.array(STARTS, dtype=np.int64), 'reference starts')
            exact(bank['target'], parent_targets[row['record_id']], 'common reference targets')
            rows.append({'record_id': row['record_id'], 'status': 'complete',
                         **old.metrics(bank['prediction'], bank['target'], common['y_scale'])})
            count += 1
        rebuilt.append({**entry, 'rows': rows})
    bla = read(paths['bla_final_npz'].parent/'evaluation.json')
    require([r['record_id'] for r in bla['rows']] == list(DEV), 'BLA12 record roster')
    bla_rows = []
    for row in bla['rows']:
        require(row['status'] == 'complete', 'closed BLA complete')
        bank = load_arrays(paths['bla_final_npz'].parent/'evaluation'/(row['record_id']+'.npz'),
            ('prediction', 'target', 'starts', 'final_state', 'singular_values',
             'context_residual_norm', 'y_context', 'u_context', 'future_u'))
        exact(bank['starts'], np.array(STARTS, dtype=np.int64), 'BLA starts')
        exact(bank['target'], parent_targets[row['record_id']], 'BLA common targets')
        bla_rows.append({'record_id': row['record_id'], 'status': 'complete',
                         **old.metrics(bank['prediction'], bank['target'], common['y_scale'])})
        count += 1
    return rebuilt, bla_rows, count


def error_record(value):
    require(isinstance(value, dict) and set(value) == {'type', 'message', 'traceback'}
            and all(isinstance(v, str) and v for v in value.values()), 'retained numerical error')


def validate_timing(row):
    keys = ('request_ms', 'copy_ms', 'initializer_ms', 'rollout_ms')
    require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in keys)
            and row['request_ms'] > 0, 'finite complete request timing')
    close(row['request_ms'], sum(row[k] for k in keys[1:]), atol=1e-8)


def closed_process(plan, registration, study, process, paths):
    terminal, launch = read(process), read(process.parent/'launch.json')
    producer = ROOT/'research/fsm_author/scripts/evaluate_nllfr_factorial.py'
    supervisor = ROOT/'research/fsm_author/scripts/run_nllfr_factorial.py'
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'), str(producer),
               '--registration', str(registration), '--output', str(study)]
    require(terminal['status'] in ('completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed')
            and terminal['end_identity_matches'] is True, 'original terminal and unchanged identity')
    require(launch['status'] == 'running' and all(terminal[k] == v for k, v in launch.items() if k != 'status'),
            'original process launch join')
    code = terminal['observed_exit_code']
    unspawned = terminal['status'] == 'supervisor_failed' and terminal.get('pid') is None
    require(type(code) is int or (unspawned and code is None), 'original exit code')
    require((terminal['status'] != 'completed' or code == 0)
            and (terminal['status'] != 'failed' or code != 0), 'terminal exit consistency')
    require(terminal['outcome'] == ('ORIGINAL_PROCESS_COMPLETE' if terminal['status'] == 'completed' else 'INCOMPLETE'),
            'terminal outcome')
    require(terminal['phase'] == 'evaluate' and terminal['command'] == command
            and terminal['registration_sha256'] == pin(registration)['sha256']
            and terminal['environment'] == old.ENV and terminal['timeout_seconds'] == 3600
            and terminal['rss_cap_bytes'] == 32*1024**3, 'registered process recipe')
    require(terminal['producer'] == descriptor(producer) and terminal['supervisor'] == descriptor(supervisor)
            and terminal['parent_fit_processes'] == {k: descriptor(paths[k+'_process']) for k in ('old', 'new')},
            'producer and closed training processes')
    for key in ('elapsed_seconds', 'wall_elapsed_seconds'):
        require(type(terminal[key]) in (int, float) and math.isfinite(terminal[key]) and terminal[key] > 0,
                'positive original duration')
    require(type(terminal['peak_polled_child_rss_bytes']) is int and terminal['peak_polled_child_rss_bytes'] >= 0,
            'original RSS receipt')
    log = process.parent/'process.log'
    require(terminal['log_sha256'] == (pin(log)['sha256'] if log.exists() else None), 'original process log')
    files = old.inventory(study) if study.exists() else {}
    require(terminal['artifacts'] == files, 'closed exact full artifact inventory')
    if terminal['status'] == 'completed':
        started = read(study/'started.json')
        require(started == {'registration_sha256': pin(registration)['sha256'], 'pid': terminal['pid']}, 'original child identity')
        for name, digest in plan['source_sha256'].items():
            require(files['source/'+name]['sha256'] == digest, 'frozen source snapshot')
    return terminal, files


def authenticate(study, process, freeze):
    """Opaque metadata admission before any new numerical array is decoded."""
    study, process, freeze = Path(study).resolve(), Path(process).resolve(), Path(freeze).resolve()
    held, registration = source_freeze(freeze)
    plan = read(registration)
    require(set(plan) == {'version', 'experiment', 'source_sha256', 'prerequisites', 'output', 'process_directory'}
            and plan['version'] == VERSION and plan['experiment'] == EXPERIMENT, 'fixed four-cell registration')
    require(study == ROOT/old.relative(plan['output'])
            and process == ROOT/old.relative(plan['process_directory'])/'process.json', 'registered original paths')
    require(MIN_SOURCES <= set(plan['source_sha256']) and REQUIRED <= set(plan['prerequisites']), 'required source/input roster')
    for name, digest in plan['source_sha256'].items():
        require(pin(ROOT/old.relative(name))['sha256'] == digest, 'registered source drift')
    paths = {}
    for name, value in plan['prerequisites'].items():
        path = ROOT/old.relative(value['path'])
        require(pin(path)['sha256'] == value['sha256'], 'registered prerequisite drift')
        paths[name] = path
    require(read(paths['runtime_preflight'])['status'] == 'PASS', 'qualified runtime')
    qualification(plan, paths)
    parents, prior_paths = parent_admission(paths)
    terminal, files = closed_process(plan, registration, study, process, paths)
    inputs = {'registration': descriptor(registration), 'process': descriptor(process),
        'launch': descriptor(process.parent/'launch.json'), 'files': files,
        'process_log': descriptor(process.parent/'process.log') if (process.parent/'process.log').exists() else None,
        'freeze': descriptor(freeze), 'source_pins': held['sources'], 'qualification': held['qualification'],
        'prerequisites': {k: descriptor(v) for k, v in paths.items()}, 'parents': parents}
    return plan, inputs, paths, terminal, parents, prior_paths


def scored(prediction, target, scale):
    require(isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray)
            and prediction.dtype == target.dtype == np.float64 and prediction.shape == target.shape
            and prediction.ndim == 3 and prediction.shape[2] == 3 and min(prediction.shape[:2]) > 0,
            'standardized metric schema')
    finite(scale, (3,)); require((scale > 0).all(), 'positive common scale')
    with np.errstate(over='ignore', invalid='ignore'):
        square = (prediction-target)**2
        mse = float(square.mean())
        channel = np.sqrt(square.mean(axis=(0, 1)))
        native = channel*scale
    require(math.isfinite(mse) and np.isfinite(channel).all() and np.isfinite(native).all(), 'metric overflow')
    return {'rmse': math.sqrt(mse), 'mse': mse, 'per_channel_rmse': channel.tolist(),
            'native_output_per_channel_rmse': native.tolist(),
            'requests': len(prediction), 'horizon': prediction.shape[1]}


def legal_bank(value, start):
    exact(value['starts'], np.array([start], dtype=np.int64), 'request start')
    for key, shape in (('y_context', (1, 100, 3)), ('u_context', (1, 99, 3)), ('future_u', (1, 128, 3))):
        finite(value[key], shape)
    return value


def identical(a, b):
    try:
        exact(a, b)
    except ValueError:
        return False
    return True


def json_tree(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item(), allow_nan=False))


def old_parity(row, bank, saved_row, saved_bank, cache):
    if row['status'] == 'failed':
        return {'status': 'FAILED', 'reason': 'original complete request numerically failed'}
    checks = {'prediction': identical(bank['prediction'], saved_bank['prediction']),
              'context': identical(row['context'], saved_row['context']),
              'starts': identical(cache['starts'], saved_bank['starts'])}
    checks.update({k: identical(bank[k], saved_bank[k]) for k in
                   ('final_state', 'forecast_state', 'linear_seed_states', 'solved_context_start_states')})
    checks.update({k: identical(cache[k], saved_bank[k]) for k in ('y_context', 'u_context', 'future_u')})
    return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'checks': checks}


def pair_parity(short, long, a, b):
    if short['status'] != 'complete' or long['status'] != 'complete':
        return {'status': 'unavailable', 'reason': 'numerical forecast failure'}
    x, y = short['context']['requests'][0], long['context']['requests'][0]
    checks = {'same_linear_seed': identical(a['linear_seed_states'], b['linear_seed_states']),
              'same_direction_trial_prefix': identical(x['trace'], y['trace'][:len(x['trace'])]),
              'context_objective_nonincreasing': y['final_objective'] <= x['final_objective']}
    if x['status'] != 'ITERATION_CAP':
        checks['early_stop_identical'] = (identical(x, y) and all(identical(a[k], b[k]) for k in
            ('physical_prediction', 'final_state', 'forecast_state', 'solved_context_start_states')))
    return {'status': 'PASS' if all(checks.values()) else 'FAILED', 'checks': checks,
            'short_status': x['status'], 'long_status': y['status']}


def compare_numeric(saved, expected):
    require(isinstance(saved, np.ndarray) and saved.dtype == np.float64 and saved.shape == expected.shape,
            'replayed array geometry')
    # A returned forecast may have been saved before standardization/score overflow.
    exact(np.isfinite(saved), np.isfinite(expected), 'finite mask')
    exact(np.isnan(saved), np.isnan(expected), 'NaN mask')
    exact(np.isposinf(saved), np.isposinf(expected), 'positive infinity mask')
    exact(np.isneginf(saved), np.isneginf(expected), 'negative infinity mask')
    close(saved[np.isfinite(saved)], expected[np.isfinite(expected)], **TOLERANCES['replay'])


def replay_cells(study, models, common, parent, files):
    root = Path(parent['evaluation'])
    expected = set()
    collected = {c: {'rows': [], 'requests': [], 'context_status_counts': {}} for c in CELLS}
    parities = {'old16': [], 'within_checkpoint': []}
    independent, targets, complete_banks, failed = {}, {}, 0, 0
    for rid in DEV:
        pieces = {cell: [] for cell in CELLS}
        for start in STARTS:
            stem = f'{rid}/{start:04d}'
            source = root/'evaluation'/(stem+'.inputs.npz')
            name = 'requests/'+stem+'.inputs.npz'
            expected.add(name)
            require(pin(study/name) == pin(source), 'opaque original legal request copy')
            cached = legal_bank(load_arrays(study/name, INPUT_KEYS), start)
            rows, banks = {}, {}
            original_bank = None
            for cell in CELLS:
                prefix = 'cells/'+cell+'/evaluation/'+stem
                expected.add(prefix+'.json')
                row = read(study/(prefix+'.json'))
                require(row['cell'] == cell and row['record_id'] == rid and row['start'] == start
                        and row['status'] in ('complete', 'failed'), 'scheduled request identity/status')
                result, error, payload = None, None, None
                try:
                    result = replay.replay_request(models[cell[:3]], cached['y_context'],
                        cached['u_context'], cached['future_u'], iterations=int(cell[-2:]))
                except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                    if not old._numeric_failure(exc):
                        raise
                    error = exc
                if result is not None:
                    # First target/reference output read occurs after independent inference.
                    if original_bank is None:
                        original_bank = load_arrays(root/'evaluation'/(stem+'.npz'),
                            ('prediction', 'target', 'starts', 'final_state', 'forecast_state',
                             'linear_seed_states', 'solved_context_start_states', 'y_context', 'u_context', 'future_u'))
                    target = original_bank['target']
                    finite(target, (1, 128, 3))
                    targets[rid, start] = target
                    physical, final, forecast, diagnostic = result
                    with np.errstate(over='ignore', invalid='ignore'):
                        normalized = (physical-common['y_mean'])/common['y_scale']
                    expected.add(prefix+'.npz')
                    payload = load_arrays(study/(prefix+'.npz'), BANK_KEYS)
                    exact(payload['starts'], cached['starts'], 'output start')
                    exact(payload['target'], target, 'unaltered cached target')
                    for key, value in {'physical_prediction': physical, 'prediction': normalized,
                        'final_state': final, 'forecast_state': forecast,
                        'linear_seed_states': diagnostic['linear_seed_states'],
                        'solved_context_start_states': diagnostic['solved_context_start_states']}.items():
                        compare_numeric(payload[key], value)
                    close(row['context'], diagnostic, **TOLERANCES['context'])
                    validate_timing(row['diagnostic_request_timing'])
                    try:
                        score = scored(normalized, target, common['y_scale'])
                    except ValueError as exc:
                        if str(exc) != 'metric overflow':
                            raise
                        error = exc
                independent[cell, rid, start] = result
                if row['status'] == 'failed':
                    require(error is not None, 'reported failure independently reproduced')
                    error_record(row['error'])
                    require(set(row) == {'cell', 'record_id', 'start', 'status', 'error'}
                            | ({'context', 'diagnostic_request_timing'} if result is not None else set()),
                            'failed attempt evidence schema')
                    failed += 1
                else:
                    require(error is None and payload is not None, 'successful numerical replay')
                    close({k: row[k] for k in score}, scored(payload['prediction'], payload['target'], common['y_scale']))
                    require(set(row) == {'cell', 'record_id', 'start', 'status', 'context',
                            'diagnostic_request_timing', *score}, 'successful attempt schema')
                    pieces[cell].append(payload)
                    counts = collected[cell]['context_status_counts']
                    status = row['context']['requests'][0]['status']
                    counts[status] = counts.get(status, 0)+1
                    complete_banks += 1
                rows[cell], banks[cell] = row, payload
                collected[cell]['requests'].append({k: row[k] for k in ('cell', 'record_id', 'start', 'status')})
                if cell == 'old16':
                    if row['status'] == 'complete':
                        prior_row = read(root/'evaluation'/(stem+'.json'))
                        check = old_parity(row, payload, prior_row, original_bank, cached)
                    else:
                        check = {'status': 'FAILED', 'reason': 'original complete request numerically failed'}
                    parities['old16'].append({'record_id': rid, 'start': start, **check})
            for model in ('old', 'new'):
                a, b = model+'16', model+'64'
                check = pair_parity(rows[a], rows[b], banks[a], banks[b])
                if check['status'] == 'PASS':
                    compare_budgets(rows[a]['context'], rows[b]['context'], banks[a], banks[b])
                parities['within_checkpoint'].append({'checkpoint': model, 'record_id': rid, 'start': start, **check})
        for cell in CELLS:
            parts = pieces[cell]
            row = {'record_id': rid, 'status': 'complete' if len(parts) == 32 else 'incomplete',
                   'completed_requests': len(parts), 'expected_requests': 32}
            if len(parts) == 32:
                name = 'cells/'+cell+'/evaluation/'+rid+'.npz'
                expected.add(name)
                bank = load_arrays(study/name, BANK_KEYS)
                for key in BANK_KEYS:
                    exact(bank[key], np.concatenate([part[key] for part in parts]), 'all32 request aggregation')
                try:
                    row.update(scored(bank['prediction'], bank['target'], common['y_scale']))
                except ValueError as exc:
                    if str(exc) != 'metric overflow':
                        raise
                    row['status'] = 'incomplete'
            collected[cell]['rows'].append(row)
    # Missing targets are decoded only after all four attempts at every slot.
    target_banks = {}
    for rid in DEV:
        values = []
        for start in STARTS:
            if (rid, start) not in targets:
                values.append(load_arrays(root/'evaluation'/rid/f'{start:04d}.npz',
                    ('prediction', 'target', 'starts', 'final_state', 'forecast_state', 'linear_seed_states',
                     'solved_context_start_states', 'y_context', 'u_context', 'future_u'))['target'])
            else:
                values.append(targets[rid, start])
        target_banks[rid] = np.concatenate(values)
    require(expected <= set(files), 'required attempt and bank inventory')
    return collected, parities, independent, target_banks, expected, complete_banks, failed


def evaluator_identity(before, after, plan, preflight):
    require(before == after, 'runtime/source identity closure')
    environment = before['thread_environment']
    require(set(environment) == {*old.ENV, 'XLA_FLAGS'}
            and all(environment[k] == value for k, value in old.ENV.items())
            and (environment['XLA_FLAGS'] is None or isinstance(environment['XLA_FLAGS'], str)),
            'qualified evaluator runtime environment')
    # The qualified evaluator records XLA_FLAGS informationally; before/after
    # equality above closes that value without inventing a new runtime setting.
    require(before['source_sha256'] == plan['source_sha256'] and before['prerequisites'] == plan['prerequisites']
            and before['python'] == preflight['python'] and before['executable'] == preflight['executable']
            and before['versions'] == preflight['versions'] and before['direct_url'] == preflight['upstream_direct_url']
            and before['installed_author_sha256'] == {k: v['sha256'] for k, v in preflight['installed_source_matches'].items()},
            'qualified evaluator runtime')


def audit(study, process, freeze):
    study = Path(study).resolve()
    plan, inputs, paths, terminal, parent, reference_paths = authenticate(study, process, freeze)
    result = {'status': 'PASS', 'agreement': True, 'study': str(study), 'inputs': inputs,
        'scientific_status': 'REFERENCE_INCOMPLETE', 'tolerances': TOLERANCES,
        'scope': 'Independent pinned NumPy context replay and saved normalized-bank scoring. No producer/model imports, '
                 'optimizer calls, raw archive/recording decoding or timing replay. Parent admission may hash opaque archives.'}
    counts = {'request_replays': 0, 'reference_banks_rescored': 0, 'timing_replays': 0,
              'optimizer_calls': 0, 'raw_archive_decodes': 0, 'new_recording_decodes': 0}
    if terminal['status'] != 'completed':
        result.update(counts=counts, results={'matrix_status': 'FACTORIAL_INCOMPLETE',
            'original_stop': terminal['status'], 'continuation': None})
        require(authenticate(study, process, freeze)[1] == inputs, 'partial evidence closure')
        return result
    expected = {'started.json', 'identity-before.json', 'identity-after.json', 'admission.json', 'events.jsonl',
        'old-final.npz', 'new-final.npz', 'common-normalizer.npz', 'references.json', 'evaluation.json', 'summary.json'}
    expected.update('source/'+name for name in plan['source_sha256'])
    require(read(study/'admission.json') == parent, 'same original metadata admission')
    before = read(study/'identity-before.json')
    evaluator_identity(before, read(study/'identity-after.json'), plan, read(paths['runtime_preflight']))
    for key, name in (('old_final', 'old-final.npz'), ('new_final', 'new-final.npz'),
                      ('common_normalizer', 'common-normalizer.npz')):
        require(pin(study/name) == pin(paths[key]), 'original admitted numeric copy')
    # All provenance joins above precede the first numerical decode.
    models = {name: load_arrays(study/(name+'-final.npz'), old.NAMES) for name in ('old', 'new')}
    for model in models.values():
        old.validate_model(model)
    common = load_arrays(study/'common-normalizer.npz', ('u_mean', 'u_scale', 'y_mean', 'y_scale'))
    for value in common.values():
        finite(value, (3,))
    require((common['y_scale'] > 0).all() and (common['u_scale'] > 0).all(), 'positive common normalization')
    cells, parity, independent, targets, roster, bank_count, failures = replay_cells(
        study, models, common, parent, inputs['files'])
    expected.update(roster)
    evaluation = read(study/'evaluation.json')
    require(set(evaluation) == {'cells', 'timing', 'parity', 'parity_pass', 'contrasts', 'prior_continuation', 'continuation'}
            and set(evaluation['cells']) == set(CELLS), 'exact four-cell evaluation schema')
    exact(evaluation['parity'], parity, 'all384 original and768 paired parity results')
    parity_ok = all(row['status'] == 'PASS' for group in parity.values() for row in group)
    require(evaluation['parity_pass'] is parity_ok, 'parity outcome')
    medians, timing_complete = timing_summary(evaluation['timing'])
    for row in (*evaluation['timing']['warmups'], *evaluation['timing']['samples']):
        replayed = independent[row['cell'], row['record_id'], row['start']]
        require((row['status'] == 'complete') == (replayed is not None), 'timed inference outcome versus independent same request')
        if replayed is not None:
            require(row['error'] is None, 'complete timing has no error')
            close(row['context'], replayed[3], **TOLERANCES['context'])
    for cell in CELLS:
        saved, computed = evaluation['cells'][cell], cells[cell]
        for a, b in zip(saved['rows'], computed['rows'], strict=True):
            if 'error' in a:
                require(b['status'] == 'incomplete' and b['completed_requests'] == 32, 'aggregate failure only after32 requests')
                error_record(a['error'])
                close({k: v for k, v in a.items() if k != 'error'}, b)
            else:
                close(a, b)
        exact(saved['requests'], computed['requests'], 'full request roster')
        exact(saved['context_status_counts'], computed['context_status_counts'], 'all context stop counts')
        computed['mean_rmse'] = aggregate(computed['rows'])
        computed['median_request_ms'] = medians[cell]
        computed['amplitude_mean_rmse'] = {}
        for amplitude in ('100mV', '200mV'):
            own_rows = [r for r in computed['rows'] if r['record_id'].startswith(amplitude)]
            computed['amplitude_mean_rmse'][amplitude] = (float(np.mean([r['rmse'] for r in own_rows]))
                if len(own_rows) == 6 and all(r['status'] == 'complete' for r in own_rows) else None)
        computed['model_numeric_bytes'] = sum(v.nbytes for v in models[cell[:3]].values())
        computed['state_bytes'], computed['policy_bytes'] = 28*8, 9*8
        computed['persistent_numeric_bytes'] = sum(computed[k] for k in ('model_numeric_bytes', 'state_bytes', 'policy_bytes'))
        computed['failure_counts'] = {'forecast': sum(r['status'] != 'complete' for r in computed['requests']),
            'record': sum(r['status'] != 'complete' for r in computed['rows']),
            'warmup': sum(r['status'] != 'complete' for r in evaluation['timing']['warmups'] if r['cell'] == cell),
            'timing': sum(r['status'] != 'complete' for r in evaluation['timing']['samples'] if r['cell'] == cell)}
        for key in ('mean_rmse', 'amplitude_mean_rmse', 'median_request_ms', 'model_numeric_bytes',
                    'state_bytes', 'policy_bytes', 'persistent_numeric_bytes', 'failure_counts'):
            close(saved[key], computed[key])
        require(saved['storage_scope'] == 'All19 numeric model arrays, retained state and nine C/H/solver-policy scalars; no prepared cache', 'full numeric storage scope')
    references, bla_rows, reference_count = reference_scores(reference_paths, common, targets)
    validate_reference_report(read(study/'references.json'), parent, references, bla_rows, cells)
    matrix_complete = parity_ok and timing_complete and all(cells[c]['mean_rmse'] is not None for c in CELLS)
    decision = decisions(references, bla_rows, cells, fit_complete=parent['new_fit_status'] == 'FIT_ONLY_COMPLETE',
                         matrix_complete=matrix_complete, parity_pass=parity_ok)
    effects = contrasts({c: cells[c]['mean_rmse'] for c in CELLS})
    close(evaluation['continuation'], decision)
    close(evaluation['contrasts'], effects)
    exact(evaluation['prior_continuation'], parent['prior_continuation'], 'unchanged historical scalar report')
    summary = read(study/'summary.json')
    require(set(summary) == {'status', 'new_fit_status', 'new_fit_iterations', 'cells', 'counts', 'continuation', 'contrasts', 'scope'}
            and set(summary['cells']) == set(CELLS), 'exact four-cell summary schema')
    require(summary['status'] == ('FACTORIAL_COMPLETE' if matrix_complete else 'FACTORIAL_INCOMPLETE')
            and summary['new_fit_status'] == parent['new_fit_status']
            and summary['new_fit_iterations'] == parent['new_fit_iterations'], 'matrix versus training status')
    close(summary['contrasts'], effects); close(summary['continuation'], decision)
    require(summary['counts'] == {'forecast_attempts': 1536, 'record_slots': 48, 'warmups': 4, 'timed_requests': 96}, 'all scheduled counts')
    for cell in CELLS:
        close(summary['cells'][cell], {k: cells[cell][k] for k in
              ('mean_rmse', 'amplitude_mean_rmse', 'median_request_ms', 'persistent_numeric_bytes', 'context_status_counts', 'failure_counts')})
    events = [json.loads(line) for line in (study/'events.jsonl').read_text().splitlines()]
    require([r['phase'] for r in events] == ['record_closed']*12+['evaluation_closed']
            and all(a['time_ns'] <= b['time_ns'] for a, b in pairwise(events)),
            'original evaluation event chronology')
    for event, rid in zip(events[:12], DEV, strict=True):
        require(event['record_id'] == rid and event['cells'] ==
                {c: next(r['status'] for r in cells[c]['rows'] if r['record_id'] == rid) for c in CELLS},
                'all cell record closure events')
    close({k: events[-1][k] for k in summary}, summary)
    require(set(inputs['files']) == expected, 'exact dynamic output inventory')
    require(authenticate(study, process, freeze)[1] == inputs, 'all inputs/sources/output bytes unchanged')
    counts.update(request_replays=1536, record_slots=48, complete_request_banks=bank_count,
        failed_request_replays=failures, reference_banks_rescored=reference_count, warmups=4, timing_slots=96)
    result.update(scientific_status='REFERENCE_COMPLETE' if decision['reference_complete'] else 'REFERENCE_INCOMPLETE',
        counts=counts, results={'matrix_status': summary['status'], 'cells': cells, 'contrasts': effects,
        'continuation': decision, 'new_fit_status': parent['new_fit_status'],
        'new_fit_iterations': parent['new_fit_iterations'], 'parity': parity})
    return result



def validate_reference_report(saved, parent, references, bla_rows, cells):
    exact(saved['banks'], parent['reference_banks'], 'unchanged 312 bank identities')
    exact(saved['historical_continuation'], parent['prior_continuation'], 'unchanged historical report')
    rows = [{**row, 'family': e['family'], 'seed': e['seed']} for e in references for row in e['rows']]
    rows.extend({**row, 'family': old.BLA, 'seed': None} for row in bla_rows)
    close(saved['rows'], rows)
    fresh = decisions(references, bla_rows, cells, fit_complete=False, matrix_complete=False)
    maps = {'family_means', 'per_record_means', 'per_seed_means', 'per_amplitude_means'}
    expected = {**parent['prior_continuation'], **{key: fresh[key] for key in maps}}
    close(saved['rescored_continuation'], expected)
    require(saved['scope'] == '312 unchanged normalized prediction/target banks rescored; no model calls or historical report edits',
            'reference scoring scope')


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'process', 'freeze', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive independent audit output')
    result = audit(args.study, args.process, args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'scientific_status': result['scientific_status'],
                      'counts': result['counts']}))


if __name__ == '__main__':
    main()
