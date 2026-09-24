"""Publication from closed, independently audited convex-readout JSON only.

No array/checkpoint decoding, model construction, generation or solve occurs.
All registered sources, original process closures, inventories and upstream
proof bytes are authenticated before consuming the current study metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from publish_finite_convex_readout_stop import authenticate as authenticate_stopped
from render_finite_factorized_dynamics import authenticate as authenticate_upstream

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-gap-readout-study-report-v1'
STUDY_NAME = 'finite-gap-readout-study-v1'
PARENTS = ('factorized', 'matched_free', 'dense_free')
HEADS = ('original', 'solved')
ARMS = tuple(parent + '_' + head for parent in PARENTS for head in HEADS)
SEEDS = (426261001, 426261002, 426261003)
HORIZONS = (1, 2, 4, 8)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')
CONFIG = {'dev_seed_namespace': 428260924, 'dev_attempts': 128, 'dev_horizon': 8,
          'batch_size': 64, 'parent_seeds': list(SEEDS), 'parent_arms': list(PARENTS),
          'train_namespace': 426260924, 'train_horizon': 2,
          'solver_maxiter': 20000, 'solver_certificate_every': 10,
          'solver_monotonicity_roundoff': 1e-15, 'solver_gap_tolerance': 1e-8,
          'solver_nonincrease_tolerance': 1e-12}
ROUTES = ('train_blind', 'train_observed',
          *(head + '_' + route for head in HEADS for route in ('blind', 'observed', 'shuffled')))
METHOD_REGISTRATION_SHA = '3513fd540b1a01af6568a4268a8ea7e0a93680ad39ef35ddb206d998cbf255eb'
PREDECESSOR_REGISTRATION_SHA = '129af3b025548eda6e475cb2d69bd5a7f3bfb325b085a56dff3ee71af798bab4'
FAILED_INTEGRATION_SHA = '3bb2b18fe3f264b3843454a974c381fdda2646507c75fd32697f0aa8e6acf795'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve() == path,
            'absolute regular original file: ' + str(path))
    return path


def descriptor(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'original phase directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no symlinks in original output')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(path)
    return result


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'finite publication scalar')
    elif type(value) is dict:
        for child in value.values():
            finite_tree(child)
    elif type(value) is list:
        for child in value:
            finite_tree(child)


def closed_phase(plan, plan_path, phase):
    spec = plan['phases'][phase]
    folder, launch_path = Path(spec['output']), Path(spec['supervision'])
    require(launch_path.name.endswith('.launch.json'), 'original launch suffix')
    terminal_path = launch_path.with_name(launch_path.name[:-len('.launch.json')] + '.terminal.json')
    receipt_path = Path(str(folder) + '.receipt.json')
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(receipt['phase'] == phase and receipt['status'] == 'PASS'
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == descriptor(plan_path)['sha256']
            and receipt['supervision'] == str(launch_path) and receipt['output'] == str(folder), 'original registered successful phase')
    require(receipt['sources_before'] == receipt['sources_after'] == plan['sources'], 'unchanged phase source closure')
    require(receipt['launch'] == launch and all(terminal[key] == value for key, value in launch.items()),
            'actual original launch equals receipt and terminal')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['group_absent'] is True and terminal['timed_out'] is False
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True, 'successful original process closure')
    require(terminal['cwd'] == str(ROOT) and terminal['cap_seconds'] == spec['cap_seconds']
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + spec['cap_seconds'] * 10**9
            and math.isclose(terminal['wall_seconds'], (terminal['finished_ns'] - terminal['started_ns']) / 1e9,
                             rel_tol=0, abs_tol=1e-9), 'complete registered native-clock timing')
    command = terminal['command']
    expected_flags = {'--plan': str(plan_path), '--plan-sha256': descriptor(plan_path)['sha256'],
                      '--phase': phase, '--supervision': str(launch_path), '--output': str(folder)}
    require(len(command) == 2 + 2 * len(expected_flags) and command[0] == plan['runtime']['executable']
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_gap_readout_study_worker_v2.py'
            and len(set(command[2::2])) == len(expected_flags), 'exact original worker and unique flags')
    arguments = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(arguments) == set(expected_flags), 'exact registered worker flag names')
    for flag in ('--plan', '--supervision', '--output'):
        arguments[flag] = str((ROOT / arguments[flag]).resolve())
    require(arguments == expected_flags, 'registered original worker arguments')
    require(terminal['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and terminal['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'source-bound original supervisor and clock')
    require(inventory(folder) == receipt['files'], 'all original phase payloads unchanged')
    return {'directory': folder, 'receipt': receipt, 'terminal': terminal,
            'receipt_path': receipt_path, 'terminal_path': terminal_path, 'launch_path': launch_path}


def authenticate_failed_integration(plan):
    """Authenticate the preserved failed first integration without promoting it."""
    proof = plan['engineering_predecessor']
    study = ROOT / 'output' / STUDY_NAME
    old_path = study / 'engineering-registration-01.json'
    require(proof['status'] == 'FAILED' and proof['folder'] == str(study / 'engineering-source-01')
            and proof['registration'] == {'path': str(old_path), **descriptor(old_path)}
            and descriptor(old_path)['sha256'] == FAILED_INTEGRATION_SHA, 'fixed failed integration identity')
    old = read(old_path)
    require(old['version'] == STUDY_NAME and old['mode'] == 'engineering' and old['root'] == str(ROOT)
            and old['config'] == CONFIG and len(old['sources']) == 55
            and inventory(Path(proof['folder'])) == proof['files']
            and set(proof['files']) == set(old['sources']) | {'manifest.json'}
            and {name: proof['files'][name] for name in old['sources']} == old['sources'],
            'all original failed-attempt sources preserved separately')
    require(read(Path(proof['folder']) / 'manifest.json') == {
        'registration_sha256': FAILED_INTEGRATION_SHA, 'sources': old['sources']},
        'original failed-source snapshot manifest joins its registration')
    for name, pin in old['sources'].items():
        require(plan['sources'].get(name) == pin and descriptor(ROOT / name) == pin, 'failed original source remains immutable')
    spec = old['phases']['qualify']
    folder, launch_path = Path(spec['output']), Path(spec['supervision'])
    require(folder == study / 'engineering-01' and launch_path == study / 'engineering-native-01.launch.json',
            'original failed integration paths')
    paths = {'receipt': Path(str(folder) + '.receipt.json'), 'launch': launch_path,
             'terminal': study / 'engineering-native-01.terminal.json', 'log': study / 'engineering-native-01.log'}
    for key, path in paths.items():
        require(proof[key] == {'path': str(path), **descriptor(path)}, 'failed integration original ' + key)
    receipt, launch, terminal = (read(paths[name]) for name in ('receipt', 'launch', 'terminal'))
    require(receipt['phase'] == 'qualify' and receipt['status'] == 'FAILED'
            and receipt['plan'] == str(old_path) and receipt['plan_sha256'] == FAILED_INTEGRATION_SHA
            and receipt['output'] == str(folder) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == old['sources'] and receipt['launch'] == launch
            and all(terminal[key] == value for key, value in launch.items()), 'original failed integration joins')
    require(terminal['status'] == 'failed' and terminal['returncode'] == 1 and terminal['timed_out'] is False
            and terminal['group_absent'] is True and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == []
            and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
            'failed tests closed normally with cleanup')
    require(terminal['cwd'] == str(ROOT) and terminal['cap_seconds'] == spec['cap_seconds'] == 300
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + 300 * 10**9
            and math.isclose(terminal['wall_seconds'], (terminal['finished_ns'] - terminal['started_ns']) / 1e9,
                             rel_tol=0, abs_tol=1e-9), 'original failed integration native timing')
    command = terminal['command']
    expected = {'--plan': str(old_path), '--plan-sha256': FAILED_INTEGRATION_SHA, '--phase': 'qualify',
                '--supervision': str(launch_path), '--output': str(folder)}
    require(len(command) == 12 and command[0] == old['runtime']['executable']
            and (ROOT / command[1]).resolve() == ROOT / 'scripts/finite_gap_readout_study_worker.py'
            and len(set(command[2::2])) == 5, 'original failed v1 worker')
    observed = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(observed) == set(expected), 'original failed worker flag roster')
    for flag in ('--plan', '--supervision', '--output'):
        observed[flag] = str((ROOT / observed[flag]).resolve())
    require(observed == expected
            and terminal['watchdog_sha256'] == old['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and terminal['clock_source_sha256'] == old['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'failed original command and supervisor pins')
    expected_commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *old['lint_sources']],
                         [old['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *old['tests']]]
    require(len(receipt['commands']) == 2 and [row['command'] for row in receipt['commands']] == expected_commands
            and [row['returncode'] for row in receipt['commands']] == [0, 1]
            and inventory(folder) == receipt['files'], 'original failed tests and all log bytes')
    text = regular(folder / 'command-1.log').read_text()
    require(re.search(r'\b1 failed, 74 passed\b', text) is not None, 'original 74-pass/one-failure outcome')
    return {'proof': proof, 'status': 'FAILED', 'tests_passed': 74, 'tests_failed': 1,
            'wall_seconds': terminal['wall_seconds'], 'finished_ns': terminal['finished_ns'],
            'scope': 'The first integration rejected an audit guard test before scientific execution. Its original source snapshot, logs and failure remain separate evidence.'}


def authenticate_prerequisites(plan):
    """Join the exact closed synthetic qualification and failed predecessor."""
    expected_folders = {'method_qualification': ROOT / 'output/finite-gap-solver-qualification-v1',
                        'predecessor': ROOT / 'output/finite-convex-readout-v1'}
    for key, folder in expected_folders.items():
        record = plan[key]
        require(record['folder'] == str(folder) and inventory(folder) == record['files'], 'pinned complete history: ' + key)
        path = regular(Path(record['registration']['path']))
        require(record['registration'] == {'path': str(path), **descriptor(path)}, 'original history registration descriptor')
    stopped_path, _stopped_plan, _stopped_q, _stopped_phases = authenticate_stopped(
        expected_folders['predecessor'], PREDECESSOR_REGISTRATION_SHA)
    require(plan['predecessor']['registration']['path'] == str(stopped_path)
            and plan['predecessor']['status'] == 'STOPPED_BEFORE_DEV', 'failed predecessor remains stopped')
    proof = plan['method_qualification']
    registration = regular(Path(proof['registration']['path']))
    require(registration == expected_folders['method_qualification'] / 'registration.json'
            and descriptor(registration)['sha256'] == METHOD_REGISTRATION_SHA, 'fixed numerical prerequisite registration')
    source = read(registration)
    require(source['version'] == 'finite-gap-solver-qualification-v1' and source['root'] == str(ROOT)
            and source['empirical_inputs'] == [] and source['successor_admitted'] is False
            and source['cap_seconds'] == 600, 'synthetic-only original qualification scope')
    for name, pin in source['sources'].items():
        require(plan['sources'].get(name) == pin and descriptor(ROOT / name) == pin, 'qualified method source closure')
    folder, launch_path = Path(source['output']), regular(Path(source['supervision']))
    receipt_path = regular(Path(str(folder) + '.receipt.json'))
    terminal_path = launch_path.with_name(launch_path.name.removesuffix('.launch.json') + '.terminal.json')
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    require(proof['receipt'] == {'path': str(receipt_path), **descriptor(receipt_path)}
            and proof['terminal'] == {'path': str(terminal_path), **descriptor(terminal_path)}, 'original prerequisite receipt and terminal pins')
    require(receipt['version'] == source['version'] and receipt['status'] == 'COMPLETED'
            and receipt['engineering_gate'] == 'PASS' and receipt['qualifying_fixtures'] == {'gap_projected': 18, 'slsqp': 10}
            and receipt['plan'] == str(registration) and receipt['plan_sha256'] == METHOD_REGISTRATION_SHA
            and receipt['output'] == str(folder) and receipt['supervision'] == str(launch_path)
            and receipt['sources_before'] == receipt['sources_after'] == source['sources'], 'successful source-bound engineering prerequisite')
    require(receipt['launch'] == launch and all(terminal[key] == value for key, value in launch.items())
            and terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None and terminal['clock_error'] is None
            and terminal['timing_available'] is True, 'original prerequisite process closure')
    require(terminal['cwd'] == str(ROOT) and terminal['cap_seconds'] == 600
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + 600 * 10**9
            and math.isclose(terminal['wall_seconds'], (terminal['finished_ns'] - terminal['started_ns']) / 1e9,
                             rel_tol=0, abs_tol=1e-9), 'original prerequisite timing')
    command = terminal['command']
    expected = {'--plan': str(registration), '--plan-sha256': METHOD_REGISTRATION_SHA,
                '--supervision': str(launch_path), '--output': str(folder)}
    require(len(command) == 10 and command[0] == source['runtime']['executable']
            and (ROOT / command[1]).resolve() == ROOT / 'scripts/qualify_finite_gap_solver.py'
            and len(set(command[2::2])) == 4, 'exact original numerical-qualification worker')
    arguments = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(arguments) == set(expected), 'exact qualification arguments')
    for name in ('--plan', '--supervision', '--output'):
        arguments[name] = str((ROOT / arguments[name]).resolve())
    require(arguments == expected and terminal['watchdog_sha256'] == source['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and terminal['clock_source_sha256'] == source['sources']['src/openjev/research/suspend_clock.py']['sha256'], 'qualified original launch sources')
    expected_commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *source['lint_sources']],
                         [source['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *source['tests']]]
    require(len(receipt['commands']) == 2
            and all(row['command'] == command and row['returncode'] == 0
                    for row, command in zip(receipt['commands'], expected_commands, strict=True)), 'original lint and fabricated-test closure')
    require(inventory(folder) == receipt['files'], 'complete original synthetic payload bytes')
    return terminal['finished_ns']


def method_summary(plan):
    """Read previously authenticated synthetic JSON, without fixture decoding."""
    proof = plan['method_qualification']
    registration = read(Path(proof['registration']['path']))
    folder = Path(registration['output'])
    summary = read(folder / 'summary.json')
    require(summary['version'] == registration['version'] and summary['engineering_gate'] == 'PASS'
            and summary['qualifying_fixtures'] == {'gap_projected': 18, 'slsqp': 10}
            and summary['counts'] == {'fixtures': 18, 'gram_builds': 18, 'solver_calls': 36,
                'empirical_array_decodes': 0, 'model_calls': 0, 'checkpoint_loads': 0, 'development_generations': 0},
            'complete numerical-only qualification result')
    require(len(summary['fixtures']) == 18 and [row['name'] for row in summary['fixtures']] == registration['fixtures']
            and len(summary['results']) == 36
            and {(row['fixture'], row['method']) for row in summary['results']}
            == {(name, method) for name in registration['fixtures'] for method in ('slsqp', 'gap_projected')}, 'full shared synthetic roster')
    initial_passes = sum(row['initial_certificate']['passed'] for row in summary['fixtures'])
    require(initial_passes == 2, 'two initial-pass fixtures, shared by both methods')
    timings = {}
    for method in ('slsqp', 'gap_projected'):
        rows = sorted((row for row in summary['results'] if row['method'] == method), key=lambda row: row['execution_position'])
        require(sum(row['qualifies'] for row in rows) == summary['qualifying_fixtures'][method], 'all reported engineering outcomes retained')
        timings[method] = {'calls': len(rows), 'initially_qualifying_fixtures': initial_passes,
            'first_call_fixture': rows[0]['fixture'], 'first_call_seconds': rows[0]['solve_seconds'],
            'later_17_calls_seconds': math.fsum(row['solve_seconds'] for row in rows[1:]),
            'all_18_calls_seconds': math.fsum(row['solve_seconds'] for row in rows),
            'independent_scalar_check_seconds': math.fsum(row['independent_seconds'] for row in rows)}
    finite_tree(summary)
    return {'registration': proof['registration'], 'receipt': proof['receipt'], 'terminal': proof['terminal'],
        'engineering_gate': summary['engineering_gate'], 'qualifying_fixtures': summary['qualifying_fixtures'],
        'initially_qualifying_fixtures_per_method': initial_passes, 'timings': timings,
        'gram_build_seconds': math.fsum(row['build_seconds'] for row in summary['fixtures']),
        'fixture_generation_seconds': summary['fixture_generation_seconds'],
        'original_phase_seconds': read(Path(proof['terminal']['path']))['wall_seconds'],
        'results': summary['results'], 'counts': summary['counts'],
        'scope': 'Synthetic engineering only; no empirical solver-effectiveness or architecture claim. Two fixtures already meet the certificate at initialization.',
        'timing_scope': 'First SLSQP solve includes lazy SciPy import; there is no separately measured import-only timer. First calls and remaining calls are reported separately, without claiming matched steady-state speed. Gram construction and independent scalar checks are separate.'}


def authenticate(studyfolder):
    require(studyfolder == ROOT / 'output' / STUDY_NAME, 'exact convex-readout study directory')
    plan_path = regular(studyfolder / 'study-registration.json')
    plan = read(plan_path)
    require(plan['version'] == STUDY_NAME and plan['root'] == str(ROOT)
            and plan['mode'] == 'study' and plan['config'] == CONFIG, 'registered convex-head study')
    require({key: value['cap_seconds'] for key, value in plan['phases'].items()}
            == {'qualify': 300, 'fit': 600, 'audit': 600}, 'fixed original phase caps')
    for name, expected in plan['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(ROOT / name) == expected, 'current registered source: ' + name)
    fit, audit = (closed_phase(plan, plan_path, phase) for phase in ('fit', 'audit'))
    require(fit['terminal']['finished_ns'] <= audit['terminal']['started_ns']
            and audit['receipt']['producer_receipt'] == descriptor(fit['receipt_path'])
            and audit['receipt']['producer_terminal'] == descriptor(fit['terminal_path']),
            'original producer closes before original saved-output audit')
    q = plan['qualification']
    q_receipt_path = regular(Path(q['path']))
    require(descriptor(q_receipt_path) == q['descriptor'], 'registered qualification identity')
    q_plan_path = regular(Path(read(q_receipt_path)['plan']))
    q_plan = read(q_plan_path)
    require(q_plan['version'] == STUDY_NAME and q_plan['mode'] == 'engineering'
            and q_plan['root'] == str(ROOT) and q_plan['config'] == CONFIG
            and q_plan['sources'] == plan['sources'] and q_plan['upstream'] == plan['upstream']
            and q_plan['method_qualification'] == plan['method_qualification']
            and q_plan['engineering_predecessor'] == plan['engineering_predecessor']
            and q_plan['phases'] == plan['phases']
            and q_plan['predecessor'] == plan['predecessor'],
            'identical qualified sources, configuration and original-input inventory')
    qualify = closed_phase(q_plan, q_plan_path, 'qualify')
    require(qualify['receipt_path'] == q_receipt_path and descriptor(qualify['terminal_path']) == q['terminal']
            and qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns'],
            'qualification originally closed before producer launch')
    upstream = plan['upstream']
    upstream_folder = ROOT / 'output/finite-factorized-dynamics-v1'
    require(upstream['folder'] == str(upstream_folder)
            and inventory(upstream_folder) == upstream['files'], 'unchanged complete original parent evidence')
    prerequisite_finished = authenticate_prerequisites(plan)
    upstream_path, _upstream_plan, upstream_phases, _upstream_saved, _upstream_audit = authenticate_upstream(upstream_folder)
    require(upstream['registration'] == {'path': str(upstream_path), **descriptor(upstream_path)}
            and upstream['run'] == str(upstream_phases['fit']['directory']), 'original parent source and run identity')
    require(prerequisite_finished <= qualify['terminal']['started_ns'], 'method qualification closed before current integration qualification')
    failed = authenticate_failed_integration(plan)
    require(failed['finished_ns'] <= qualify['terminal']['started_ns'], 'failed first integration closed before corrected qualification')
    # Current result metrics are read only after all closures and pins above.
    saved, report = read(fit['directory'] / 'summary.json'), read(audit['directory'] / 'audit.json')
    require(saved['version'] == STUDY_NAME and saved['config'] == CONFIG and saved['upstream'] == upstream
            and report['version'] == 'finite-gap-readout-study-audit-v1'
            and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == report['gates'], 'successful saved audit with original external closure boundary')
    expected = {'config.json', 'train.npz', 'base.npz', 'base-prefix.npz', 'base-oracle.npz',
                'oracle-base.npz', 'oracle-base-check.json', 'predictions-base.npz',
                'solves.jsonl', 'solve-barrier.json', 'prediction-times.jsonl'}
    expected |= {f'{prefix}-{parent}-{seed}.{suffix}' for parent in PARENTS for seed in SEEDS
                 for prefix, suffix in (('states-train', 'npz'), ('states-base', 'npz'), ('solve', 'json'))}
    require(len(expected) == 38 and set(saved['files']) == expected
            and fit['receipt']['files'] == {**saved['files'], 'summary.json': descriptor(fit['directory'] / 'summary.json')}
            and set(audit['receipt']['files']) == {'audit.json'}, 'exact complete 38-payload producer plus summary and audit')
    require(fit['receipt']['result'] == {'solves': 9, 'rows': 72}
            and report['counts'] == {'array_decodes': 24, 'checkpoint_decodes': 0, 'model_calls': 0,
                'optimizer_calls': 0, 'world_or_generator_calls': 0, 'native_calls': 0}, 'exact audited work scope')
    require(saved['counts']['solver_calls'] == saved['counts']['completed_solves'] == 9
            and saved['counts']['checkpoint_decodes'] == 9 and saved['counts']['dev_generation_count'] == 1
            and saved['counts']['optimizer_steps'] == saved['counts']['checkpoint_writes'] == 0
            and saved['solve_barrier']['completed_solves'] == 9
            and saved['solve_barrier']['dev_generation_count'] == 0, 'all nine solves finish before fresh development')
    finite_tree(saved)
    finite_tree(report)
    return plan_path, plan, {'qualify': qualify, 'fit': fit, 'audit': audit}, saved, report


def extract(plan, phases, saved, report):
    rows, baselines = report['rows'], report['baseline_rows']
    pairs = {(parent, seed) for parent in PARENTS for seed in SEEDS}
    require(len(rows) == 72 and {(row['arm'], row['seed'], row['regime'], row['horizon']) for row in rows}
            == {(arm, seed, 'base', horizon) for arm in ARMS for seed in SEEDS for horizon in HORIZONS},
            'all parent/head/seed/horizon rows')
    require(len(baselines) == 4 and {(row['regime'], row['horizon']) for row in baselines}
            == {('base', horizon) for horizon in HORIZONS}, 'all uniform-reference rows')
    require(len(saved['solves']) == 9 and {(row['parent'], row['seed']) for row in saved['solves']} == pairs,
            'every frozen parent retained')
    require(len(report['certificates']) == 9
            and {(row['parent'], row['seed']) for row in report['certificates']} == pairs
            and all(row['independently_certified'] is True for row in report['certificates']),
            'all independently certified TRAIN heads')
    require(len(report['paired_comparisons']) == 36
            and {(row['parent'], row['seed'], row['horizon']) for row in report['paired_comparisons']}
            == {(parent, seed, h) for parent, seed in pairs for h in HORIZONS}, 'complete matched comparisons')
    require(len(report['frozen_invariants']) == 9
            and {(row['parent'], row['seed']) for row in report['frozen_invariants']} == pairs,
            'every frozen-state invariant retained')
    require(set(report['data_cases']) == {'train', 'base'}
            and report['data_cases']['train'] == saved['original_train_counts']['retained']
            and report['data_cases']['base'] == saved['dataset_counts']['base']['retained'], 'consistent case counts')
    for row in rows:
        require(row['cases'] == report['data_cases']['base']
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in METRICS), 'finite model cell')
    require(set(report['gates']) == set(ARMS), 'six full diagnostic groups')
    for arm in ARMS:
        require(set(report['gates'][arm]) == set(CRITERIA), 'unchanged three criteria')
        for result in report['gates'][arm].values():
            require(type(result['passed']) is bool and result['conditions']
                    and all(type(value) is bool for value in result['conditions'].values())
                    and result['passed'] == all(result['conditions'].values()), 'all gate conditions, including failures')
    lookup = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    means = [{'arm': arm, 'horizon': horizon,
              **{metric: math.fsum(lookup[arm, seed, horizon][metric] for seed in SEEDS) / 3 for metric in METRICS}}
             for arm in ARMS for horizon in (4, 8)]
    solves = {(row['parent'], row['seed']): row for row in saved['solves']}
    for row in solves.values():
        require(row['source_state_sha256'] == row['model_state_after'] and row['frozen_weights'] is True
                and row['optimizer_weight_updates'] == 0 and row['solver_result']['complete'] is True,
                'no recurrent weight update or selected failed solve')
    require(len(saved['prediction_times']) == 18
            and {(row['parent'], row['seed'], row['head']) for row in saved['prediction_times']}
            == {(parent, seed, head) for parent, seed in pairs for head in HEADS}, 'all measured evaluation views')
    for row in saved['prediction_times']:
        require(row['model_state_before'] == row['model_state_after']
                == solves[row['parent'], row['seed']]['source_state_sha256'], 'unchanged frozen model during each view')
    require(saved['structural_work'] == report['structural_work']
            and set(saved['structural_work']) == set(PARENTS), 'audited structural work')
    for work in saved['structural_work'].values():
        require(set(work) == set(ROUTES) and all(type(record) is dict and record
                and all(type(count) is int and count >= 0 for count in record.values()) for record in work.values()),
                'all 24 work routes')
    return {'version': VERSION, 'config': plan['config'], 'sources': plan['sources'],
        'rows': rows, 'baseline_rows': baselines, 'gates': report['gates'], 'three_seed_means_h4_h8': means,
        'paired_comparisons': report['paired_comparisons'], 'certificates': report['certificates'],
        'frozen_invariants': report['frozen_invariants'], 'solves': saved['solves'],
        'prediction_times': saved['prediction_times'], 'solve_barrier': saved['solve_barrier'],
        'data_cases': report['data_cases'], 'dataset_counts': {'train': saved['original_train_counts'], **saved['dataset_counts']},
        'counts': saved['counts'], 'audit_counts': report['counts'],
        'structural_work': saved['structural_work'], 'structural_work_validation': report['metadata']['structural_work_validation'],
        'independent_reconstruction': report['target_reconstruction'], 'prefix_reconstruction': report['prefix_reconstruction'],
        'oracle_checks': saved['oracle_checks'], 'generation_seconds': saved['generation_seconds'],
        'phase_seconds': {name: phase['terminal']['wall_seconds'] for name, phase in phases.items()},
        'total_phase_seconds': math.fsum(phase['terminal']['wall_seconds'] for phase in phases.values()),
        'phase_timing_scope': 'Original current-study supervisor durations including launch and cleanup; nested costs must not be added again. Prior recurrent training is excluded.',
        'mean_scope': 'Three distinct frozen parents per family, with separate decisions; not an ensemble or independent-case confidence interval.',
        'paired_difference_scope': 'Solved minus unchanged head on the same frozen checkpoint and same fresh cases; descriptive arithmetic, no additional acceptance rule.',
        'audit_limitations': report['limitations'], 'upstream': plan['upstream'],
        'method_qualification': method_summary(plan), 'predecessor': plan['predecessor'],
        'failed_integration': authenticate_failed_integration(plan),
        'architecture_claim': False, 'latent_identification_claim': False, 'native_environment_claim': False,
        'novelty_claim': False, 'model_selection': False}


def figure(numbers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors, markers = ('#666666', '#0072B2'), ('o', 's', '^')
    lookup = {(row['arm'], row['seed'], row['horizon']): row for row in numbers['rows']}
    baselines = {row['horizon']: row for row in numbers['baseline_rows']}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    for column, parent in enumerate(PARENTS):
        for row_index, metric in enumerate(('blind_regret', 'blind_cost_mse')):
            ax = axes[row_index, column]
            for head, color in zip(HEADS, colors, strict=True):
                arm = parent + '_' + head
                for seed, marker in zip(SEEDS, markers, strict=True):
                    ax.plot(HORIZONS, [lookup[arm, seed, h][metric] for h in HORIZONS],
                            color=color, marker=marker, linewidth=.8, markersize=4, alpha=.45)
                ax.plot(HORIZONS, [math.fsum(lookup[arm, seed, h][metric] for seed in SEEDS) / 3 for h in HORIZONS],
                        color=color, linewidth=2.3)
            ax.plot(HORIZONS, [baselines[h][metric] for h in HORIZONS], color='#B45B18', linestyle='--', linewidth=1)
            ax.axhline(0, color='#888888', linewidth=.6, linestyle=':')
            ax.axvspan(.85, 2.15, color='#DDE4ED', alpha=.3, zorder=0)
            ax.set_xticks(HORIZONS)
            ax.set_xlabel('Forecast horizon')
            ax.set_title(parent.replace('_', ' ') + (' | regret' if row_index == 0 else ' | cost MSE'))
            ax.grid(axis='y', alpha=.18)
    handles = [Line2D([], [], color=color, linewidth=2, label=head.capitalize() + ' head')
               for head, color in zip(HEADS, colors, strict=True)]
    handles += [Line2D([], [], color='#B45B18', linestyle='--', label='Known-dynamics uniform reference')]
    handles += [Line2D([], [], color='#555555', marker=marker, linewidth=.7, label=f'Parent seed {seed}')
                for seed, marker in zip(SEEDS, markers, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Synthetic gap-driven head diagnostic: nine frozen parents, no recurrent refitting', fontsize=14, fontweight='bold')
    fig.text(.5, .105, 'Thin lines: every parent seed. Thick lines: three-seed means. Shading: original H1/H2 TRAIN support.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.06, right=.985, top=.9, bottom=.17, hspace=.34, wspace=.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    lines = ['# Frozen recurrent states, certificate-driven decision readouts', '',
        'Nine existing recurrent checkpoints are unchanged. This diagnostic refits only their bounded linear decision heads on original H1/H2 TRAIN states, then compares original and solved heads on one fresh development pool. It is not nine new recurrent fits.', '',
        '![Every frozen parent, original and solved head, blind regret and cost MSE](benchmark.png)', '',
        '[Frozen protocol](../finite-gap-readout-study-protocol.md). Original qualification, producer and independent audit closures, all source pins and all parent evidence were authenticated before reading these metrics.', '',
        '## Unchanged absolute criteria', '', '| Parent / head | Short horizon | Blind extrapolation | Observed filtering |', '|---|---|---|---|']
    for arm in ARMS:
        cells = []
        for criterion in CRITERIA:
            gate = numbers['gates'][arm][criterion]
            cells.append(f"{'PASS' if gate['passed'] else 'FAIL'} ({sum(gate['conditions'].values())}/{len(gate['conditions'])})")
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'All applicable conditions must pass for every seed. Favorable means do not override failures. The observed-event and survival outputs are verified unchanged by the head intervention; their criteria are retained, not attributed to a new improvement.', '',
        '## All nine TRAIN numerical certificates', '',
        '| Parent | Seed | Original objective | Solved objective | FW gap | Simplex violation | Curvature bound L |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['certificates']:
        lines.append(f"| {row['parent']} | {row['seed']} | {row['objective_initial']:.9g} | {row['objective_final']:.9g} | "
                     f"{row['fw_gap']:.3g} | {row['simplex_violation']:.3g} | {row['lipschitz']:.6g} |")
    lines += ['', 'The objective is blind MSE plus observed MSE, each divided by N × 2 × 4. Absorbed zero states stay in both denominators. Each head has 32 stored probabilities, with four nonnegative values summing to one per latent column; C = 0.25 - P.', '',
        'Each attempt uses fixed-step monotone-restarted FISTA with at most 20,000 iterations and L = 2 × max row absolute sum of the Gram matrix. It checks the direct-residual certificate at iteration zero, every ten iterations and the retained final point. A quadratic increase beyond 1e-15 triggers one plain projected step; another increase fails without backtracking. No export repair occurs. Success requires simplex error at most 1e-12, signed FW gap between -1e-12 and 1e-8, and objective increase at most 1e-12. The audit reconstructs initial/final loss, gradient, gap and L from saved TRAIN states; it reconciles intermediate history and work records without replaying unsaved iterates.', '',
        'This is a float64 numerical certificate, not an interval proof. The closed simplex includes boundary heads that finite softmax logits only approach, so an improvement cannot isolate optimizer choice. Monotonicity restarts are counted within the single fixed-budget attempt. There are no new attempts, replacement parents or fallback successes.', '',
        '## Fresh-development means', '', '| Parent / head | H | Blind regret | Blind MSE | Observed MSE | Observed KL | Blind survival MAE |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_regret']:.6g} | {row['blind_cost_mse']:.6g} | "
                     f"{row['observed_cost_mse']:.6g} | {row['observed_kl']:.6g} | {row['blind_survival_mae']:.6g} |")
    lines += ['', 'Means average three separate frozen policies; they are not ensemble predictions or confidence intervals.', '',
        '## Every H4/H8 paired difference', '', 'Differences are solved minus original on the same checkpoint and cases. These are descriptive arithmetic, not new criteria or significance tests.', '',
        '| Parent | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference |', '|---|---:|---:|---:|---:|---:|']
    for row in numbers['paired_comparisons']:
        if row['horizon'] in (4, 8):
            values = row['metrics']
            lines.append(f"| {row['parent']} | {row['seed']} | {row['horizon']} | {values['blind_regret']['difference']:+.6g} | "
                         f"{values['blind_cost_mse']['difference']:+.6g} | {values['observed_cost_mse']['difference']:+.6g} |")
    lines += ['', '## Data and work', '', '| Split | Attempted prefixes | Endpoint eligible | Found-terminated prefixes | Valid prefix events |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'base'):
        row = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {row['attempted']} | {row['retained']} | {row['discarded_found']} | {row['valid_prefix_events']} |")
    lines += ['', 'TRAIN is the original namespace 426260924. DEV is fresh namespace 428260924, generated only after all nine completed solves and their durable barrier. This is further development after earlier results, not untouched final confirmation. Only surviving prefixes receive endpoint targets; first-found observations in future routes are absorbing. All parent checkpoints are retained.', '',
        '| Parent | Seed | Load seconds | TRAIN extraction seconds | Build / solve / certificate seconds | Original DEV seconds | Solved DEV seconds | Gram value calls | Iterations |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    times = {(row['parent'], row['seed'], row['head']): row['seconds'] for row in numbers['prediction_times']}
    for row in numbers['solves']:
        result = row['solver_result']
        lines.append(f"| {row['parent']} | {row['seed']} | {row['load_seconds']:.6f} | {row['extraction_seconds']:.6f} | "
                     f"{row['solve_seconds']:.6f} | {times[row['parent'], row['seed'], 'original']:.6f} | "
                     f"{times[row['parent'], row['seed'], 'solved']:.6f} | {result['work']['quadratic_value_calls']} | {result['solver']['iterations']} |")
    lines += ['', '## Counted solver work', '',
        '| Parent | Seed | Accepted iterations | Monotonicity restarts | Gram gradients | Quadratic values | Update projections | Direct residual passes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in numbers['solves']:
        work = row['solver_result']['work']
        lines.append(f"| {row['parent']} | {row['seed']} | {work['accepted_iterations']} | {work['monotonicity_restarts']} | "
                     f"{work['gradient_calls']} | {work['quadratic_value_calls']} | {work['projection_calls']} | {work['direct_objective_gradient_passes']} |")
    proof = numbers['method_qualification']
    lines += ['', 'Gram value/gradient calls use cached sufficient statistics; direct certificate passes read the complete residual problem. These operation counts are not interchangeable costs. Discarded accelerated candidates and their replacement plain steps remain counted. No projection is applied after stopping.', '',
        '## Separate synthetic engineering prerequisite', '',
        ('Before the new empirical study, the fixed 18-fixture suite qualified the gap-driven method on **18/18** cases '
         'and the unchanged SLSQP comparator on **10/18**. Each method had **two fixtures already meeting the certificate '
         'at initialization**, including small-scale or unsupported objectives. This is bounded numerical engineering evidence, '
         'not empirical decision effectiveness or an architecture comparison. All 36 outputs, including comparator failures, are retained.'), '',
        '| Method | First call seconds | Later 17 calls seconds | All 18 call seconds | Independent scalar-check seconds | Initially qualifying cases |',
        '|---|---:|---:|---:|---:|---:|']
    for method in ('slsqp', 'gap_projected'):
        timing = proof['timings'][method]
        lines.append(f"| {method} | {timing['first_call_seconds']:.6f} | {timing['later_17_calls_seconds']:.6f} | "
                     f"{timing['all_18_calls_seconds']:.6f} | {timing['independent_scalar_check_seconds']:.6f} | {timing['initially_qualifying_fixtures']} |")
    lines += ['', proof['timing_scope'], '',
        (f"The separate prerequisite supervisor ran for {proof['original_phase_seconds']:.3f} seconds. "
         f"Its 18 Gram builds took {proof['gram_build_seconds']:.6f} seconds and fixture generation took "
         f"{proof['fixture_generation_seconds']:.6f} seconds. These are nested costs, not additions to that duration. "
         'The scientific predecessor remains STOPPED_BEFORE_DEV and is preserved as failed historical evidence; '
         'this successor has a different registration and DEV namespace.'), '',
        '## Current-study timing scope', '',
        'Load time includes checkpoint decoding, model construction, strict state loading, original head export and guards. Extraction includes both TRAIN forwards and original-head validation. Solve time includes problem construction and the numerical certificate. Both evaluation views rerun frozen forwards and pay original-head computation plus explicit matrix maps; solved outputs are posthoc maps, not new checkpoint weights. Evaluation timers include all three routes and array exports, excluding writes, scalar metrics and state hashes. These are not single-decision latency or a matched speedup claim.', '',
        '| Original current-study phase | Seconds |', '|---|---:|']
    for phase in ('qualify', 'fit', 'audit'):
        lines.append(f"| {phase} | {numbers['phase_seconds'][phase]:.3f} |")
    lines += [f"| Total successful phases | {numbers['total_phase_seconds']:.3f} |", '',
        'Whole-phase durations include launch and cleanup and contain the nested timers above. Earlier recurrent training is excluded.', '',
        (f"The preserved first integration failed with {numbers['failed_integration']['tests_passed']} tests passing "
         f"and {numbers['failed_integration']['tests_failed']} failing, before scientific execution. Its original supervisor "
         f"duration was {numbers['failed_integration']['wall_seconds']:.3f} seconds, excluded from the successful-phase total. "
         'The corrected worker and separately registered second qualification do not replace that failure or its original source snapshot.'), '',
        '[summary.json](summary.json) retains all 72 metric rows, all gate conditions and failures, 36 paired comparisons, nine complete solves with original and retained final matrices, their full certificate histories, 18 evaluation records, frozen-output checks and all 24 structural-work routes. Structural counters are attested operation counts, not complete FLOPs; gradient, allocation and platform costs are not inferred from them.', '',
        '## Interpretation limits', '',
        'A numerically certified TRAIN solution only bounds this fixed-state linear-head objective. Fresh decision improvement would support a readout-fitting bottleneck under this constrained family. Failure would not prove that latent information is absent or that nonlinear readouts cannot help. The state extraction is source-qualified rather than independently rerun by the numerical audit. Event and survival equality does not identify hidden states. There is no new architecture, novelty, calibration, robotics, native-environment transfer or scenario-shift claim. All earlier verdicts remain closed.', '']
    return '\n'.join(lines)



def render(studyfolder, outputfolder):
    studyfolder, outputfolder = Path(studyfolder).resolve(), Path(outputfolder)
    require(outputfolder.is_absolute() and outputfolder.resolve() == outputfolder and not outputfolder.exists(), 'exclusive absolute report folder')
    plan_path, plan, phases, saved, report = authenticate(studyfolder)
    numbers = extract(plan, phases, saved, report)
    outputfolder.mkdir(parents=True, exist_ok=False)
    (outputfolder / 'summary.json').write_text(json.dumps(numbers, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (outputfolder / 'report.md').write_text(document(numbers))
    figure(numbers, outputfolder / 'benchmark.png')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT / name) == expected, 'registered source unchanged during publication')
    for phase in phases.values():
        require(inventory(phase['directory']) == phase['receipt']['files'], 'original results unchanged during publication')
    require(inventory(Path(plan['upstream']['folder'])) == plan['upstream']['files'], 'parent evidence unchanged during publication')
    for key in ('method_qualification', 'predecessor'):
        require(inventory(Path(plan[key]['folder'])) == plan[key]['files'], 'historical evidence unchanged during publication: ' + key)
    authenticate_failed_integration(plan)
    receipt = {'version': VERSION, 'registration': {'path': str(plan_path), **descriptor(plan_path)},
        'renderer': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'inputs': {name: {kind: {'path': str(record[kind + '_path']), **descriptor(record[kind + '_path'])}
                          for kind in ('receipt', 'launch', 'terminal')} for name, record in phases.items()},
        'method_qualification': plan['method_qualification'], 'predecessor': plan['predecessor'],
        'engineering_predecessor': plan['engineering_predecessor'],
        'sources': plan['sources'], 'files': {name: descriptor(outputfolder / name)
                                            for name in ('summary.json', 'report.md', 'benchmark.png')},
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls',
                                'optimizer_calls', 'native_calls', 'teacher_calls'), 0),
        'model_selection': False, 'result_reads_after_original_closure': True,
        'visual_review_required': True}
    (outputfolder / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.study, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
