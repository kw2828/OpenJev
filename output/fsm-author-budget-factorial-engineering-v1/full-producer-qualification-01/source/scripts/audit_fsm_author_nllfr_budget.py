# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent FIT-cache audit of a fresh-start NL-LFR budget extension.

No production numeric imports, raw archive/DEV decoding, optimizer or timing replay.
The previously qualified independent NumPy equations are reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/audit_fsm_author_nllfr.py'
HELPER_SHA = '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32'
if hashlib.sha256((ROOT/HELPER).read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('qualified independent helper changed')
_spec = importlib.util.spec_from_file_location('held_nllfr_independent_math', ROOT/HELPER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
require, read, pin, descriptor = old.require, old.read, old.pin, old.descriptor
finite, close, load_arrays = old.finite, old.close, old.load_arrays
VERSION = 'fsm-author-nllfr-budget-study-v1'
EXPERIMENT = {**old.EXPERIMENT, 'max_iter': 100000}
FIT_KEYS = ('raw_fit_u', 'raw_fit_y', 'u', 'y', 'U', 'Y', 'f_idx', 'training_x0', 'independent_x0')
TRACE_KEYS = ('loss_history', 'iter_times', 'iter_count', 'author_stop_flag', 'wall_time')
SOURCES = ('scripts/audit_fsm_author_nllfr_budget.py', 'tests/test_audit_fsm_author_nllfr_budget.py')
PARENT_FILES = {'parent_initial_zip': 'initial.zip', 'parent_initial_npz': 'initial.npz',
                'parent_original_bla': 'original-bla.npz', 'parent_fit_data': 'fit-data.npz',
                'parent_solver_trace': 'solver-trace.npz'}
REQUIRED = {'runtime_preflight', 'parent_registration', 'parent_process', 'parent_evaluation_process',
            'parent_audit', 'parent_audit_process', 'producer_qualification', 'source_review', *PARENT_FILES}
MIN_SOURCES = {'research/fsm_author/scripts/fit_nllfr_budget.py',
               'research/fsm_author/scripts/run_nllfr_budget_study.py',
               'research/fsm_author/tests/test_fit_nllfr_budget.py',
               'research/fsm_author/scripts/fit_nllfr.py', HELPER,
               'research/fsm_author/src/openjev_fsm_author/nllfr.py',
               'research/fsm_author/src/openjev_fsm_author/benchmark.py',
               'research/fsm_author/uv.lock', 'research/fsm_author/pyproject.toml'}


def exact_arrays(actual, expected, label):
    require(set(actual) == set(expected), label+' roster')
    for name, value in expected.items():
        require(isinstance(actual[name], np.ndarray) and actual[name].dtype == value.dtype
                and actual[name].shape == value.shape and np.array_equal(actual[name], value),
                label+' exact array: '+name)


def prefix_result(loss, parent, *, required=10000):
    """A loss prefix is not a certification of every intermediate parameter state."""
    require(type(required) is int and required > 0, 'positive prefix length')
    for value in (loss, parent):
        require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 1
                and len(value) > 0 and np.isfinite(value).all() and (value >= 0).all(), 'finite loss trace')
    require(len(parent) == required, 'original parent trace length')
    compared = min(len(loss), required)
    difference = loss[:compared]-parent[:compared]
    different = np.flatnonzero(loss[:compared] != parent[:compared])
    return {'required': required, 'compared': compared,
            'exact': compared == required and not len(different),
            'first_difference': int(different[0]) if len(different) else None,
            'max_absolute_difference': float(np.max(np.abs(difference)))}


def validate_trace(trace, *, cap=100000):
    require(set(trace) == set(TRACE_KEYS), 'raw trace roster')
    count, flag = trace['iter_count'], trace['author_stop_flag']
    require(count.shape == () and np.issubdtype(count.dtype, np.integer)
            and 1 <= int(count) <= cap, 'raw iteration count')
    require(flag.shape == () and flag.dtype == np.bool_, 'raw author stop flag')
    n = int(count)
    for key in ('loss_history', 'iter_times'):
        finite(trace[key], (n,))
        require((trace[key] >= 0).all(), 'nonnegative raw trace')
    finite(trace['wall_time'], ())
    require(float(trace['wall_time']) >= 0, 'nonnegative author duration')
    require(bool(flag) or n == cap, 'a normally returned nonconverged trace must reach its cap')
    return n, bool(flag)


def validate_initial(initial, parent_initial, original_bla, parent_bla, final=None):
    old.validate_model(initial)
    old.validate_model(parent_initial)
    require(set(original_bla) == set(parent_bla) == set(old.BASE), 'nested BLA roster')
    exact_arrays(initial, parent_initial, 'fresh original initialization')
    exact_arrays(original_bla, parent_bla, 'original nested BLA')
    for name in old.BASE:
        require(np.array_equal(initial[name], original_bla[name]) and initial[name].dtype == np.float64,
                'original linear start/normalizer: '+name)
    parameter_names = (*old.BASE[:4], *old.EXTRA)
    require(len(parameter_names) == 14 and sum(initial[k].size for k in parameter_names) == 7473,
            'all 7473 trainable scalars')
    if final is not None:
        old.validate_model(final)
        for name in old.BASE[4:]:
            require(np.array_equal(final[name], initial[name]), 'frozen normalization/time: '+name)
    return {name: {'changed_scalars': int(np.count_nonzero(initial[name] != final[name])),
                   'total_scalars': initial[name].size} for name in parameter_names} if final is not None else {}


def reconstruct_cache(cache, bla, *, samples=8192, realizations=6, periods=2, offset=820):
    """Derive normalization, period means, full spectrum and x0 from audited FIT cache."""
    require(set(cache) == set(FIT_KEYS), 'FIT-cache roster')
    rebuilt = {}
    for name in ('u', 'y'):
        raw = cache['raw_fit_'+name]
        finite(raw, (samples, 3, realizations, periods))
        mean = raw.mean(axis=(0, 2, 3), keepdims=True)
        std = raw.std(axis=(0, 2, 3), keepdims=True)
        require(np.isfinite(std).all() and (std > 0).all(), 'positive FIT-only scales')
        close(bla[name+'_mean'], mean.reshape(3), **old.TOLERANCES['preprocessing'])
        close(bla[name+'_std'], std.reshape(3), **old.TOLERANCES['preprocessing'])
        values = (raw-mean)/std
        rebuilt[name] = values.mean(axis=3)
        rebuilt[name.upper()] = np.fft.rfft(values, axis=0).mean(axis=3)
        for key in (name, name.upper()):
            finite(cache[key], rebuilt[key].shape, complex_value=key.isupper())
            close(cache[key], rebuilt[key], **old.TOLERANCES['preprocessing'])
    require(cache['f_idx'].dtype == np.int64 and np.array_equal(cache['f_idx'], np.arange(1, 3840)),
            'unchanged excited-bin metadata')
    x0 = old.periodic_x0(bla, rebuilt['U'], samples=samples, offset=offset)
    for key in ('training_x0', 'independent_x0'):
        finite(cache[key], x0.shape)
        close(cache[key], x0, **old.TOLERANCES['preprocessing'])
    return rebuilt, x0


def scientific_status(author_stop, prefix_exact, terminal_complete=True):
    require(type(author_stop) is bool and type(prefix_exact) is bool and type(terminal_complete) is bool,
            'explicit scientific status booleans')
    return 'FIT_ONLY_COMPLETE' if author_stop and prefix_exact and terminal_complete else 'FIT_ONLY_INCOMPLETE'


def bound(value):
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
            and descriptor(value['path']) == value, 'bound evidence descriptor')
    return Path(value['path'])


def source_freeze(freeze):
    held = read(freeze)
    require(held['status'] == 'FROZEN_BEFORE_EMPIRICAL_AUDIT', 'independent audit source freeze')
    require(set(held['sources']) == {*SOURCES, HELPER}, 'independent audit source roster')
    for name, value in held['sources'].items():
        require(bound(value) == ROOT/name, 'current frozen audit source')
    require(held['sources'][HELPER]['sha256'] == HELPER_SHA, 'held independent arithmetic')
    qualification = read(bound(held['qualification']))
    expected = {k: held['sources'][k] for k in SOURCES}
    require(qualification['status'] == 'PASS' and qualification['sources_unchanged'] is True
            and qualification['sources_before'] == qualification['sources_after'] == expected,
            'passing original fabricated qualification')
    commands = [['.venv/bin/ruff', 'check', *SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', SOURCES[1]]]
    require(len(qualification['commands']) == 2, 'qualification command count')
    for row, command in zip(qualification['commands'], commands, strict=True):
        require(row['command'] == command and row['returncode'] == 0, 'qualified original command')
        bound(row['log'])
    preflight = read(bound(qualification['preflight']))
    require(preflight['sources'] == expected and preflight['commands'] == commands,
            'qualification preflight joins')
    require(set(preflight['snapshots']) == set(SOURCES), 'qualified snapshot roster')
    for name, value in preflight['snapshots'].items():
        bound(value)
        require(value['sha256'] == expected[name]['sha256'], 'qualified source snapshot')
    return held, bound(held['registration'])


def parent_lineage(paths):
    """Admit the already-audited FIT cache; never invoke old raw/DEV admission."""
    parent = read(paths['parent_audit'])
    process = read(paths['parent_process'])
    closure = read(paths['parent_audit_process'])
    require(parent['status'] == 'PASS' and parent['agreement'] is True
            and parent['results']['fit_status'] == 'iteration_cap_reached', 'audited capped parent')
    inputs = parent['inputs']
    require(inputs['registration'] == descriptor(paths['parent_registration'])
            and inputs['process'] == descriptor(paths['parent_process'])
            and inputs['evaluation_process'] == descriptor(paths['parent_evaluation_process'])
            and inputs['source'] == descriptor(ROOT/HELPER), 'parent audited identity')
    require(pin(paths['parent_registration'])['sha256'] == old.REGISTRATION_SHA256,
            'original parent registration')
    require(process['status'] == 'completed' and process['observed_exit_code'] == 0
            and process['end_identity_matches'] is True and process['identity_error'] is None
            and process['registration_sha256'] == old.REGISTRATION_SHA256,
            'original parent process closure')
    require(closure['state'] == 'EXITED' and closure['observed_exit_code'] == 0
            and closure['success'] is True and closure['sources_unchanged'] is True
            and closure['inputs_unchanged'] is True and closure['error'] is None
            and closure['closure_error'] is None and closure['audit_output'] == descriptor(paths['parent_audit'])
            and closure['fit_process'] == inputs['process']
            and closure['evaluation_process'] == inputs['evaluation_process'], 'original parent audit terminal')
    require(closure['inputs_before'] == closure['inputs_after']
            and closure['sources_before'] == closure['sources_after'], 'original parent audit identity closure')
    for value in (*closure['sources_before'].values(), *closure['inputs_before'].values(), closure['log']):
        bound(value)
    require(closure['sources_before'][HELPER] == descriptor(ROOT/HELPER), 'independent parent source')
    folder = Path(parent['study'])
    require(folder == ROOT/'output/fsm-author-nllfr-study-v1', 'original parent study role')
    for key, name in PARENT_FILES.items():
        require(paths[key] == folder/name and pin(paths[key]) == inputs['files'][name]
                and process['artifacts'][name] == inputs['files'][name], 'audited parent input: '+key)
    require(pin(folder/'fit.json') == inputs['files']['fit.json'], 'parent fit ledger pin')
    fit = read(folder/'fit.json')
    require(fit['status'] == 'iteration_cap_reached' and fit['iterations'] == 10000
            and fit['author_stop_flag'] is False, 'original 10000-loss capped trace')
    return parent, fit


def closed_process(plan, registration, study, process, paths):
    terminal = read(process)
    launch = read(process.parent/'launch.json')
    command = [str(ROOT/'research/fsm_author/.venv/bin/python'),
               str(ROOT/'research/fsm_author/scripts/fit_nllfr_budget.py'),
               '--registration', str(registration), '--output', str(study)]
    require(terminal['phase'] == 'fit' and terminal['command'] == command
            and terminal['registration_sha256'] == pin(registration)['sha256'], 'registered original command')
    require(terminal['status'] in ('completed', 'failed', 'timeout', 'memory_limit', 'supervisor_failed')
            and terminal['end_identity_matches'] is True and terminal['identity_error'] is None,
            'closed original process with source closure')
    require(launch['status'] == 'running'
            and all(terminal[k] == v for k, v in launch.items() if k != 'status'), 'original launch join')
    code = terminal['observed_exit_code']
    not_spawned = terminal['status'] == 'supervisor_failed' and terminal.get('pid') is None
    require(code is None if not_spawned else type(code) is int and type(terminal['pid']) is int,
            'terminal child identity')
    require((terminal['status'] != 'completed' or code == 0)
            and (terminal['status'] != 'failed' or code != 0), 'terminal exit consistency')
    require(terminal['outcome'] == ('ORIGINAL_PROCESS_COMPLETE' if terminal['status'] == 'completed' else 'INCOMPLETE'),
            'original completion status')
    require(terminal['timeout_seconds'] == 18000 and terminal['rss_cap_bytes'] == 32*1024**3
            and terminal['environment'] == old.ENV, 'registered resource/environment limits')
    for key in ('elapsed_seconds', 'wall_elapsed_seconds'):
        require(type(terminal[key]) in (int, float) and math.isfinite(terminal[key]) and terminal[key] > 0,
                'original terminal duration')
    require(type(terminal['peak_polled_child_rss_bytes']) is int and terminal['peak_polled_child_rss_bytes'] >= 0,
            'sampled memory evidence')
    require(terminal['parent_fit_process'] == descriptor(paths['parent_process']), 'fresh-start parent process join')
    for key, name in (('producer', 'fit_nllfr_budget.py'), ('supervisor', 'run_nllfr_budget_study.py')):
        require(terminal[key] == descriptor(ROOT/'research/fsm_author/scripts'/name), 'original launcher source')
    log = process.parent/'process.log'
    require((pin(log)['sha256'] if log.exists() else None) == terminal['log_sha256']
            and (log.exists() or not_spawned), 'original terminal log')
    files = old.inventory(study) if study.exists() else {}
    require(files == terminal['artifacts'], 'exact original artifact inventory')
    if 'started.json' in files:
        started = read(study/'started.json')
        require(started['pid'] == terminal['pid'] and started['registration'] == descriptor(registration),
                'original child start identity')
    if terminal['status'] == 'completed':
        for name, digest in plan['source_sha256'].items():
            require(files['source/'+name]['sha256'] == digest, 'original source snapshot')
    return terminal, files


def authenticate(study, process, freeze):
    study, process, freeze = (Path(p).resolve() for p in (study, process, freeze))
    held, registration = source_freeze(freeze)
    plan = read(registration)
    require(set(plan) == {'version', 'experiment', 'source_sha256', 'prerequisites', 'output', 'process_directory'}
            and plan['version'] == VERSION and plan['experiment'] == EXPERIMENT, 'FIT-only registered recipe')
    require(study == ROOT/old.relative(plan['output'])
            and process == ROOT/old.relative(plan['process_directory'])/'process.json', 'registered original paths')
    require(REQUIRED <= set(plan['prerequisites']), 'all parent/qualification prerequisites')
    require(MIN_SOURCES <= set(plan['source_sha256']), 'required implementation source roster')
    for name, digest in plan['source_sha256'].items():
        require(pin(ROOT/old.relative(name))['sha256'] == digest, 'registered source drift')
    paths = {}
    for key, value in plan['prerequisites'].items():
        paths[key] = ROOT/old.relative(value['path'])
        require(pin(paths[key])['sha256'] == value['sha256'], 'registered prerequisite drift: '+key)
    for key in ('runtime_preflight', 'producer_qualification', 'source_review'):
        require(read(paths[key])['status'] == 'PASS', 'failed prerequisite: '+key)
    qualification = read(paths['producer_qualification'])
    definition = read(bound(qualification['definition']))
    require(qualification['sources_before'] == qualification['sources_after'] == definition['sources']
            and set(definition['sources']) == set(plan['source_sha256']), 'producer qualification source closure')
    for name, value in definition['sources'].items():
        require(bound(value) == ROOT/old.relative(name)
                and value['sha256'] == plan['source_sha256'][name], 'producer qualified implementation')
    require([r['command'] for r in qualification['commands']] == definition['commands'],
            'producer original qualification commands')
    for row in qualification['commands']:
        require(row['returncode'] == 0, 'producer original qualification failure')
        bound(row['log'])
    review = read(paths['source_review'])
    require(review['qualification'] == plan['prerequisites']['producer_qualification']
            and review['reviewed_source_sha256'] == plan['source_sha256'], 'producer source-review identity')
    parent, parent_fit = parent_lineage(paths)
    terminal, files = closed_process(plan, registration, study, process, paths)
    inputs = {'registration': descriptor(registration), 'process': descriptor(process),
              'launch': descriptor(process.parent/'launch.json'), 'files': files,
              'process_log': descriptor(process.parent/'process.log') if (process.parent/'process.log').exists() else None,
              'freeze': descriptor(freeze), 'qualification': held['qualification'], 'source_pins': held['sources'],
              'prerequisites': {k: descriptor(p) for k, p in paths.items()}}
    return plan, inputs, paths, terminal, parent, parent_fit


def validate_fit(fit, trace, prefix, initial_pin, final_pin, prefix_pin):
    n, stopped = validate_trace(trace)
    optimizer = 'complete' if stopped else 'iteration_cap_reached'
    expected = optimizer if prefix['exact'] else 'prefix_mismatch'
    require(fit['iterations'] == n and fit['author_stop_flag'] is stopped
            and fit['optimizer_status'] == optimizer and fit['status'] == expected
            and fit['budget_only_attributable'] is prefix['exact'], 'fit/prefix completion semantics')
    require(fit['initial'] == initial_pin and fit['final'] == final_pin and fit['prefix'] == prefix_pin
            and fit['trainable_scalars'] == 7473 and fit['discarded_vendor_warmup_steps'] == 1
            and fit['zip_roundtrip_all_numeric_arrays_bitwise_equal'] is True, 'fit identity/counts')
    close(fit['author_reported_seconds'], float(trace['wall_time']))
    for key in ('optimization_and_preservation_compile_inclusive_seconds', 'author_reported_seconds'):
        require(type(fit[key]) in (int, float) and math.isfinite(fit[key]) and fit[key] >= 0, 'saved training duration')
    require(type(fit['peak_ru_maxrss_bytes']) is int and fit['peak_ru_maxrss_bytes'] >= 0, 'saved peak memory')
    return scientific_status(stopped, prefix['exact'])


def audit(study, process, freeze):
    study = Path(study).resolve()
    plan, inputs, paths, terminal, _parent, parent_fit = authenticate(study, process, freeze)
    counts = {'original_fit_attempts': 1, 'raw_archive_decodes': 0, 'dev_decodes': 0,
              'model_or_context_forecasts': 0, 'native_objective_replays': 0, 'periodic_bla_solves': 0,
              'optimizer_calls': 0, 'backward_calls': 0, 'timing_replays': 0}
    result = {'status': 'PASS', 'agreement': True, 'scientific_status': 'FIT_ONLY_INCOMPLETE',
              'study': str(study), 'inputs': inputs, 'counts': counts,
              'tolerances': old.TOLERANCES,
              'scope': 'Independent NumPy FIT-cache objective replay only. No raw archive or DEV decode, optimizer, '
                       'gradient, timing or context replay. Exact loss prefix is not an intermediate-weight certificate.'}
    if terminal['status'] != 'completed':
        result['results'] = {'original_stop': terminal['status'], 'budget_only_attributable': False,
                             'fit_status': 'original_process_incomplete'}
        require(authenticate(study, process, freeze)[1] == inputs, 'partial evidence closure')
        return result
    expected = {'started.json', 'identity-before.json', 'identity-after.json', 'admission.json', 'events.jsonl',
                'initial.zip', 'initial.npz', 'initial-loaded.npz', 'original-bla.npz', 'fit-data.npz',
                'parent-solver-trace.npz', 'fresh-start.json', 'final.zip', 'final.npz', 'solver-trace.npz',
                'prefix.json', 'roundtrip.npz', 'roundtrip-original-bla.npz', 'serialization.json', 'fit.json',
                'summary.json', *('source/'+name for name in plan['source_sha256'])}
    require(set(inputs['files']) == expected, 'exact complete budget-fit inventory')
    identity = read(study/'identity-before.json')
    require(identity == read(study/'identity-after.json'), 'original runtime/source identity closure')
    old.check_runtime(identity, plan, read(paths['runtime_preflight']))
    for key, name in PARENT_FILES.items():
        destination = 'parent-solver-trace.npz' if key == 'parent_solver_trace' else name
        require(pin(study/destination) == pin(paths[key]), 'exact original parent bytes copied')
    admission = read(study/'admission.json')
    require(admission['input_files'] == {k: descriptor(paths[k]) for k in PARENT_FILES}
            and admission['raw_archive_decodes'] == admission['dev_evaluation_calls'] == 0
            and admission['fit_records'] == 12, 'FIT-cache-only numerical admission')
    initial = load_arrays(study/'initial.npz', old.NAMES)
    final = load_arrays(study/'final.npz', old.NAMES)
    parent_initial = load_arrays(paths['parent_initial_npz'], old.NAMES)
    bla = load_arrays(study/'original-bla.npz', old.BASE)
    parameter_changes = validate_initial(initial, parent_initial, bla,
                                        load_arrays(paths['parent_original_bla'], old.BASE), final)
    exact_arrays(load_arrays(study/'initial-loaded.npz', old.NAMES), initial, 'initial ZIP numeric export')
    cache = load_arrays(study/'fit-data.npz', FIT_KEYS)
    derived, x0 = reconstruct_cache(cache, bla)
    counts['periodic_bla_solves'] = 4097
    objectives = [old.native_objective(m, derived['u'], derived['Y'], x0) for m in (initial, final)]
    counts['native_objective_replays'] = 2
    close(objectives[0], parent_fit['initial_fit_loss'], **old.TOLERANCES['objective'])
    fresh = read(study/'fresh-start.json')
    require(fresh['original_initial_zip'] == descriptor(paths['parent_initial_zip'])
            and fresh['original_initial_arrays'] == descriptor(paths['parent_initial_npz'])
            and fresh['all_initial_arrays_exact'] is fresh['original_bla_exact'] is True
            and fresh['fit_normalization_spectra_and_x0_exact'] is True
            and fresh['trainable_leaves'] == 14 and fresh['trainable_scalars'] == 7473
            and fresh['optimizer_initialization'] == 'fresh unchanged BFGS'
            and fresh['parent_final_loaded'] is False
            and fresh['dev_evaluation_calls'] == fresh['model_selection_calls'] == 0, 'fresh-start scope')
    close(fresh['initial_fit_loss'], objectives[0], **old.TOLERANCES['objective'])
    parent_trace = load_arrays(study/'parent-solver-trace.npz', TRACE_KEYS)
    require(validate_trace(parent_trace, cap=10000) == (10000, False), 'original capped trace')
    trace = load_arrays(study/'solver-trace.npz', TRACE_KEYS)
    validate_trace(trace)
    prefix = prefix_result(trace['loss_history'], parent_trace['loss_history'])
    require(read(study/'prefix.json') == prefix, 'independent exact loss-prefix reconstruction')
    fit = read(study/'fit.json')
    scientific = validate_fit(fit, trace, prefix, descriptor(study/'initial.npz'),
                              descriptor(study/'final.npz'), descriptor(study/'prefix.json'))
    for key, value in zip(('initial_fit_loss', 'final_fit_loss'), objectives, strict=True):
        close(fit[key], value, **old.TOLERANCES['objective'])
    require(objectives[1] <= objectives[0], 'returned FIT objective worsening')
    exact_arrays(load_arrays(study/'roundtrip.npz', old.NAMES), final, 'final ZIP numeric export')
    exact_arrays(load_arrays(study/'roundtrip-original-bla.npz', old.BASE), bla, 'final nested BLA export')
    serialization = read(study/'serialization.json')
    require(serialization['all_live_arrays_bitwise_equal'] is serialization['original_bla_arrays_bitwise_equal'] is True
            and serialization['files'] == {n: descriptor(study/n) for n in ('final.zip', 'final.npz',
                 'original-bla.npz', 'roundtrip.npz', 'roundtrip-original-bla.npz')}, 'serialization evidence')
    summary = read(study/'summary.json')
    require(summary['status'] == scientific and summary['fit'] == fit
            and summary['dev_evaluation_calls'] == summary['model_selection_calls'] == summary['raw_archive_decodes'] == 0,
            'FIT-only summary without reference promotion')
    events = [json.loads(line) for line in (study/'events.jsonl').read_text().splitlines()]
    require([r['phase'] for r in events] == ['optimization_started', 'fit_closed']
            and events[0]['time_ns'] <= events[1]['time_ns'] and events[0]['max_iter'] == 100000
            and events[1]['status'] == scientific and events[1]['iterations'] == fit['iterations']
            and events[1]['prefix_exact'] is prefix['exact'], 'original chronology/counters')
    close(events[0]['initial_fit_loss'], objectives[0], **old.TOLERANCES['objective'])
    require(authenticate(study, process, freeze)[1] == inputs, 'all source/input/output pins unchanged')
    counts.update(compared_loss_prefix=prefix['compared'], trainable_scalars=7473, trainable_leaves=14,
                  fit_records=12, iterations=fit['iterations'])
    result.update(scientific_status=scientific, parameter_changes=parameter_changes,
        results={'fit_status': fit['status'], 'optimizer_status': fit['optimizer_status'],
                 'budget_only_attributable': prefix['exact'], 'prefix': prefix,
                 'native_objectives': objectives, 'author_stop_flag': fit['author_stop_flag'],
                 'iterations': fit['iterations'],
                 'serialization_scope': 'Producer ZIP export equality checked numerically; ZIP loader not replayed here.'})
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'process', 'freeze', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive audit output')
    result = audit(args.study, args.process, args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'scientific_status': result['scientific_status'],
                      'counts': result['counts']}))


if __name__ == '__main__':
    main()
