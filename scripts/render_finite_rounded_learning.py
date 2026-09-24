"""Publish only closed, authenticated JSON for rounded-transition learning.

No model, generator, optimizer or array decoder is imported or called. Original
process closures and complete opaque inventories precede any result read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-rounded-learning-report-v1'
STUDY_NAME = 'finite-rounded-learning-v1'
REGISTRATION_SHA256 = '9779c1fa2db7ef6a69c195793b7ca7f0da992c55c9433c441a563d4c6d614a45'
KERNEL_REGISTRATION_SHA256 = '0922b1ecc92016b42410ef652d87f2a534be0b64b3750bff977947322e8ae375'
KERNEL_PUBLICATION_SHA256 = '1970b5920d5cb3b239bc2d2ddf79c99dc9280361195c5a9c403bb8bf9a9b78b7'
ARMS = ('original_free', 'matched_free', 'rounded')
LABELS = ('Original free', 'Matched free', 'Rounded transport')
SEEDS = (432261001, 432261002, 432261003)
HORIZONS = (1, 2, 4, 8)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')
CONFIG = {'seed_namespace': 432260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2,
          'dev_horizon': 8, 'stage1_seconds': 10., 'total_seconds': 40.}


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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_rounded_learning_worker.py'
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
    log_path = regular(launch_path.with_name(launch_path.name[:-len('.launch.json')] + '.log'))
    return {'directory': folder, 'receipt': receipt, 'terminal': terminal,
            'receipt_path': receipt_path, 'terminal_path': terminal_path, 'launch_path': launch_path,
            'log_path': log_path, 'log_descriptor': descriptor(log_path)}


def producer_files():
    names = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
             'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl',
             'training-orders.jsonl', 'checkpoint-barrier.json', 'prediction-times.jsonl',
             'oracle-train-check.json', 'oracle-base-check.json', 'train-prefix.npz', 'base-prefix.npz'}
    for arm in ARMS:
        for seed in SEEDS:
            names |= {f'allocation-{arm}-{seed}.json', f'prefix-{arm}-{seed}-base.npz'}
            for label in ('initial', 'boundary', 'final'):
                names |= {f'{label}-{arm}-{seed}.npz', f'{label}-optimizer-{arm}-{seed}.json'}
    require(len(names) == 88, 'complete fixed producer inventory')
    return names


def authenticate(studyfolder):
    """Return closed original records for this renderer and its packager."""
    studyfolder = Path(studyfolder)
    require(studyfolder == ROOT / 'output' / STUDY_NAME, 'exact original study directory')
    plan_path = regular(studyfolder / 'study-registration.json')
    require(descriptor(plan_path)['sha256'] == REGISTRATION_SHA256, 'exact immutable scientific registration')
    plan = read(plan_path)
    require(plan['version'] == STUDY_NAME and plan['mode'] == 'study'
            and plan['root'] == str(ROOT) and plan['config'] == CONFIG
            and len(plan['sources']) == 63, 'fixed study, configuration and source roster')
    require({phase: spec['cap_seconds'] for phase, spec in plan['phases'].items()}
            == {'qualify': 300, 'fit': 1200, 'audit': 600}
            and plan['rss_limit_bytes'] == 4 * 1024**3
            and plan['output_limit_bytes'] == 512 * 1024**2, 'registered resource limits')
    for name, expected in plan['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(ROOT / name) == expected, 'unchanged registered source: ' + name)
    # This pinned helper imports only standard-library metadata/clock support.
    import finite_rounded_learning_worker as worker

    proof = plan['kernel_qualification']
    require(Path(proof['folder']) == ROOT / 'output/finite-rounded-transition-qualification-v1'
            and inventory(Path(proof['folder'])) == proof['files'], 'complete unchanged numerical prerequisite')
    kernel_folder = Path(proof['folder'])
    kernel_plan_path = kernel_folder / 'registration-01.json'
    require(descriptor(kernel_plan_path)['sha256'] == KERNEL_REGISTRATION_SHA256,
            'exact originally qualified rounded primitive')
    kernel_plan = read(kernel_plan_path)
    kernel_receipt_path = kernel_folder / 'engineering-01.receipt.json'
    kernel = read(kernel_receipt_path)
    require(kernel_plan['version'] == 'finite-rounded-transition-qualification-v1'
            and kernel_plan['root'] == str(ROOT) and kernel['status'] == 'PASS'
            and kernel['plan'] == str(kernel_plan_path)
            and kernel['plan_sha256'] == KERNEL_REGISTRATION_SHA256
            and kernel['sources_before'] == kernel['sources_after'] == kernel_plan['sources']
            and kernel['supervision'] == str(kernel_folder / 'native-01.launch.json')
            and kernel['output'] == str(kernel_folder / 'engineering-01')
            and kernel['launch'] == read(Path(kernel['supervision']))
            and inventory(Path(kernel['output'])) == kernel['files'], 'original successful numerical prerequisite')
    for name, pin in kernel_plan['sources'].items():
        require(plan['sources'].get(name) == pin == descriptor(kernel_folder / 'source-snapshot-01' / name),
                'same qualified primitive and original snapshot')
    kernel_terminal = read(worker.closed_producer(kernel))
    require(all(kernel_terminal[key] == value for key, value in kernel['launch'].items())
            and kernel_terminal['clock_error'] is None and kernel_terminal['cleanup']['group_absent'] is True,
            'full original primitive process closure')
    require(kernel['launch']['command'] == [kernel_plan['runtime']['executable'],
            str(ROOT / 'scripts/qualify_finite_rounded_transition.py'), 'worker',
            '--plan', str(kernel_plan_path), '--plan-sha256', KERNEL_REGISTRATION_SHA256,
            '--supervision', kernel['supervision'], '--output', kernel['output']]
            and kernel['launch']['cwd'] == str(ROOT) and kernel['launch']['cap_seconds'] == 90
            and [row['command'] for row in kernel['commands']] == kernel_plan['commands']
            and len(kernel['commands']) == 2 and all(row['returncode'] == 0 for row in kernel['commands']),
            'original bounded primitive commands, not a rerun')
    kernel_publication = Path(proof['publication'])
    require(kernel_publication == ROOT / 'research/finite-rounded-transition-qualification-results'
            and inventory(kernel_publication) == proof['publication_files']
            and descriptor(kernel_publication / 'receipt.json')['sha256'] == KERNEL_PUBLICATION_SHA256,
            'exact opaque primitive publication')
    published_kernel = read(kernel_publication / 'receipt.json')
    require(published_kernel['status'] == 'PASS' and published_kernel['archive_roundtrip'] is True
            and published_kernel['numerical_calls'] == published_kernel['array_decodes'] == 0
            and proof['files'] == published_kernel['inputs']
            and proof['publication_files'] == {**published_kernel['outputs'],
                'receipt.json': descriptor(kernel_publication / 'receipt.json')}, 'primitive publication evidence joins')
    qualification = plan['qualification']
    q_path = regular(Path(qualification['path']))
    require(descriptor(q_path) == qualification['descriptor'], 'original qualification receipt identity')
    q_terminal = worker.admit_qualification(plan, q_path)
    require(descriptor(q_terminal) == qualification['terminal'], 'original qualification terminal identity')
    q_receipt = read(q_path)
    q_plan_path = regular(Path(q_receipt['plan']))
    qualify = closed_phase(read(q_plan_path), q_plan_path, 'qualify')
    fit = closed_phase(plan, plan_path, 'fit')
    audit = closed_phase(plan, plan_path, 'audit')
    require(qualify['receipt_path'] == q_path and qualify['terminal_path'] == q_terminal
            and qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns']
            and fit['terminal']['finished_ns'] <= audit['terminal']['started_ns'], 'ordered original phase closures')
    require(audit['receipt']['producer_receipt'] == descriptor(fit['receipt_path'])
            and audit['receipt']['producer_terminal'] == descriptor(fit['terminal_path']), 'audit authenticates this closed producer')
    require(set(fit['receipt']['files']) == producer_files() | {'summary.json'}
            and set(audit['receipt']['files']) == {'audit.json'}
            and set(qualify['receipt']['files']) == {'command-0.log', 'command-1.log'}, 'exact original phase payload rosters')
    commands = qualify['receipt']['commands']
    require(len(commands) == 2 and all(r['returncode'] == 0 for r in commands)
            and commands[0]['command'] == [str(ROOT / '.venv/bin/ruff'), 'check', *plan['lint_sources']]
            and commands[1]['command'] == [plan['runtime']['executable'], '-m', 'pytest', '-q',
                                           '-p', 'no:cacheprovider', '--noconftest', *plan['tests']], 'exact source-qualified engineering commands')
    # No fit or audit result JSON is opened before all joins above succeed.
    saved = read(fit['directory'] / 'summary.json')
    report = read(audit['directory'] / 'audit.json')
    require(saved['version'] == STUDY_NAME and saved['config'] == CONFIG
            and report['version'] == 'finite-rounded-learning-audit-v1'
            and report['profile'] == 'science' and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == {'gates': report['gates'], 'advance': report['advance']}, 'authoritative independent saved-output audit')
    require(set(saved['files']) == producer_files()
            and fit['receipt']['files'] == {**saved['files'], 'summary.json': descriptor(fit['directory'] / 'summary.json')}
            and fit['receipt']['result'] == {'fits': 9, 'rows': 36}
            and report['counts'] == {'array_decodes': 45, 'checkpoint_decodes': 27, 'optimizer_json_decodes': 27, 'model_calls': 0,
                'optimizer_calls': 0, 'world_or_generator_calls': 0, 'native_calls': 0}, 'complete artifacts and audit scope')
    finite_tree(saved)
    finite_tree(report)
    phases = {'qualify': qualify, 'fit': fit, 'audit': audit}
    inputs = {str(plan_path): descriptor(plan_path), str(q_plan_path): descriptor(q_plan_path)}
    for record in phases.values():
        for kind in ('receipt', 'launch', 'terminal'):
            path = record[kind + '_path']
            inputs[str(path)] = descriptor(path)
        inputs[str(record['log_path'])] = record['log_descriptor']
        for name, value in record['receipt']['files'].items():
            inputs[str(record['directory'] / name)] = value
    inputs.update({str(ROOT / name): value for name, value in plan['sources'].items()})
    inputs.update({str(Path(proof['folder']) / name): value for name, value in proof['files'].items()})
    inputs.update({str(kernel_publication / name): value for name, value in proof['publication_files'].items()})
    return {'plan': plan, 'plan_path': plan_path, 'phases': phases, 'summary': saved,
            'audit': report, 'inputs': inputs}


def extract(auth):
    saved, audit = auth['summary'], auth['audit']
    fits = {(r['arm'], r['seed']): r for r in saved['fits']}
    pairs = {(a, s) for a in ARMS for s in SEEDS}
    require(len(saved['fits']) == len(fits) == 9 and set(fits) == pairs, 'complete nine fits')
    rows = audit['rows']
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in rows}
    require(len(rows) == len(keyed) == 36 and set(keyed) == {(a, s, h) for a, s in pairs for h in HORIZONS},
            'every fitted endpoint row')
    require(set(audit['gates']) == set(ARMS), 'three original criterion families')
    for family in audit['gates'].values():
        require(set(family) == set(CRITERIA), 'all three original criteria')
        for gate in family.values():
            require(gate['passed'] is all(gate['conditions'].values()), 'unchanged condition conjunction')
    advance = audit['advance']
    require(advance['name'] == 'ROUNDED_ADVANCE' and len(advance['conditions']) == 21
            and all(type(value) is bool for value in advance['conditions'].values())
            and advance['passed'] is all(advance['conditions'].values()), 'complete rounded advance rule')
    boundary_records = audit['allocation']['boundary_transition_diagnostics']
    boundary_keys = {(r['arm'], r['seed'], r['boundary']) for r in boundary_records}
    require(len(boundary_records) == len(boundary_keys) == 27 and boundary_keys
            == {(a, s, b) for a, s in pairs for b in ('initial', 'boundary', 'final')},
            'all original, stage-one and final transition diagnoses')
    rounded_records = []
    for row in boundary_records:
        if row['arm'] == 'rounded':
            rounding = row['rounding']
            require(row['transition_map'] == 'four_sweep_slack_rounding'
                    and row['correction_status'] == 'reconstructed' and type(rounding) is dict
                    and rounding['admitted'] is True and rounding['sweeps'] == 4
                    and rounding['slack'] == 1e-8 and rounding['tolerance'] == 1e-12
                    and row['row_residual_max'] <= 1e-12 and row['column_residual_max'] <= 1e-12
                    and len(rounding['correction_mass']) == 4, 'actual audited rounded boundary')
            rounded_records.append(row)
        else:
            require(row['transition_map'] == 'column_softmax' and row['correction_status'] == 'not_executed'
                    and row['rounding'] is None, 'free controls receive no invented rounding diagnosis')
    require(len(rounded_records) == 9, 'all nine rounded boundaries')
    allocation_records, costs = [], []
    for arm in ARMS:
        for seed in SEEDS:
            fit = fits[arm, seed]
            path = auth['phases']['fit']['directory'] / fit['allocation']['path']
            require(descriptor(path) == {k: fit['allocation'][k] for k in ('sha256', 'bytes')},
                    'authenticated allocation JSON')
            allocation = read(path)
            require(allocation['arm'] == 'prefix_then_joint' and allocation['status'] == 'PASS'
                    and allocation['total_seconds'] == 40. and allocation['stage1_seconds'] == 10.
                    and allocation['initial_model_sha256'] == fit['initial_state_sha256']
                    and allocation['final_model_sha256'] == fit['final_state_sha256'], 'exact retained allocation')
            require(fit['parameter_metadata']['parameter_count'] == 352, 'same small model')
            finite_tree(allocation)
            allocation_records.append({'arm': arm, 'seed': seed, 'schedule': 'prefix_then_joint', 'artifact': fit['allocation'],
                **{k: allocation[k] for k in ('attempted_updates', 'accepted_updates', 'accepted_joint_updates',
                    'accepted_prefix_updates', 'joint_cursor', 'stages', 'boundary', 'checkpoints', 'work',
                    'update_work', 'timed_seconds', 'overrun_seconds', 'final_summary_seconds', 'final_summary_work')}})
            costs.append({'arm': arm, 'seed': seed, 'seconds': fit['seconds'],
                          'timed_seconds': allocation['timed_seconds'],
                          'summary_seconds': allocation['final_summary_seconds'],
                          'overrun_seconds': allocation['overrun_seconds'],
                          'accepted_joint': allocation['accepted_joint_updates'],
                          'accepted_prefix': allocation['accepted_prefix_updates'],
                          'attempted': allocation['attempted_updates'],
                          'construction_seconds': fit['construction_seconds'],
                          'construction_work': fit['construction_work']})
    means = [{'arm': arm, 'horizon': h, **{m: math.fsum(keyed[arm, s, h][m] for s in SEEDS) / 3 for m in METRICS}}
             for arm in ARMS for h in HORIZONS]
    log = (auth['phases']['qualify']['directory'] / 'command-1.log').read_text()
    matches = re.findall(r'(?m)^\s*(\d+) passed(?:, [^\n]+)? in [^\n]+$', log)
    require(len(matches) == 1 and int(matches[0]) == 216 and re.search(r'216 passed, 1 warning in ', log),
            'one original 216-pass, one-warning qualification')
    return {'version': VERSION, 'config': CONFIG, 'gates': audit['gates'], 'advance': advance,
            'rows': rows, 'three_seed_means': means, 'baseline_rows': audit['baseline_rows'],
            'allocation_comparisons': audit['allocation_comparisons'], 'training_costs': costs,
            'allocations': allocation_records, 'fits': saved['fits'], 'prefix_rows': audit['prefix_rows'],
            'boundary_transition_diagnostics': boundary_records, 'rounded_boundary_diagnostics': rounded_records,
            'boundary_checks': audit['allocation']['boundary_checks'],
            'initial_function_checks': [row['initial_transition_check']
                for row in audit['allocation']['boundary_checks'] if row['arm'] == 'original_free'],
            'dataset_counts': saved['dataset_counts'], 'scientific_counts': saved['counts'],
            'audit_counts': audit['counts'], 'structural_work': saved['structural_work'],
            'prediction_times': saved['prediction_times'], 'prefix_prediction_times': saved['prefix_prediction_times'],
            'generation_seconds': saved['generation_seconds'], 'engineering_tests_passed': int(matches[0]),
            'engineering_test_warnings': 1,
            'phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
            'total_supervised_seconds': math.fsum(v['terminal']['wall_seconds'] for v in auth['phases'].values()),
            'limitations': audit['limitations'], 'model_selection': False, 'architecture_claim': False,
            'mean_scope': 'Arithmetic means of separately fitted models on common cases, not an ensemble or significance test.',
            'timing_scope': 'Fit seconds include construction, both allocation stages, discarded attempts, boundary checkpoints, final diagnostics and durable full trace; final fit-row publication and shared preprocessing remain included in outer producer time.'}


def figure(n, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors, markers = ('#666666', '#0072B2', '#D55E00'), ('o', 's', '^')
    rows = {(r['arm'], r['seed'], r['horizon']): r for r in n['rows']}
    times = {(r['arm'], r['seed']): r['seconds'] for r in n['training_costs']}
    reference = {r['horizon']: r['blind_regret'] for r in n['baseline_rows']}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9))
    for axis, metric, title, ylabel in (
            (axes[0], 'blind_regret', 'Decision regret without new observations', 'Expected cost regret'),
            (axes[1], 'observed_kl', 'Filtering with intervening observations', 'Outcome KL divergence')):
        for arm, color in zip(ARMS, colors, strict=True):
            for seed, marker in zip(SEEDS, markers, strict=True):
                axis.plot(HORIZONS, [rows[arm, seed, h][metric] for h in HORIZONS],
                          color=color, marker=marker, linewidth=.8, markersize=4, alpha=.42)
            axis.plot(HORIZONS, [math.fsum(rows[arm, seed, h][metric] for seed in SEEDS) / 3 for h in HORIZONS],
                      color=color, linewidth=2.2)
        axis.axvspan(.8, 2.2, color='#DDE4ED', alpha=.3, zorder=0)
        axis.set(title=title, xlabel='Forecast horizon', ylabel=ylabel, xticks=HORIZONS)
    axes[0].plot(HORIZONS, [reference[h] for h in HORIZONS], color='#111111', linestyle='--')
    for seed, marker in zip(SEEDS, markers, strict=True):
        values = [times[a, seed] for a in ARMS]
        axes[2].plot(range(3), values, color='#888888', linewidth=.7, alpha=.45)
        for i, color in enumerate(colors):
            axes[2].scatter(i, values[i], color=color, marker=marker, s=40)
    for i, arm in enumerate(ARMS):
        mean = math.fsum(times[arm, seed] for seed in SEEDS) / 3
        axes[2].plot((i - .12, i + .12), (mean, mean), color=colors[i], linewidth=3)
        axes[2].annotate(f'{mean:.3f} s', (i, mean), xytext=(0, 12),
                         textcoords='offset points', ha='center', color=colors[i], fontsize=9)
    axes[2].axhline(40., color='#111111', linestyle=':', linewidth=1)
    axes[2].set_xticks(range(3), ('Original\nfree', 'Matched\nfree', 'Rounded'))
    axes[2].set(title='Measured full training cost', ylabel='Seconds per fit', xlim=(-.4, 2.4))
    for axis in axes:
        axis.grid(axis='y', alpha=.18)
        axis.set_ylim(bottom=0)
    axes[2].set_ylim(0, max(times.values()) * 1.13)
    handles = [Line2D([], [], color=c, linewidth=2, label=label) for c, label in zip(colors, LABELS, strict=True)]
    handles += [Line2D([], [], color='#111111', linestyle='--', label='Known-dynamics uniform reference (regret)')]
    handles += [Line2D([], [], color='#777777', marker=m, linewidth=.7, label=f'Seed {seed}')
                for seed, m in zip(SEEDS, markers, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    passed = n['advance']['passed']
    count = sum(n['advance']['conditions'].values())
    fig.suptitle(f"Rounded transition: {'PASS' if passed else 'FAIL'} on the registered advance rule ({count}/21)",
                 fontsize=14, fontweight='bold')
    fig.text(.5, .14, 'All nine fits shown. Thick marks are means, not an ensemble. Shading: H1/H2 supervision. Costs include overruns.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, top=.85, bottom=.25, wspace=.33)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(n):
    advance = n['advance']
    lines = ['# Rounded transition under one training-time allowance', '',
        f"**{advance['status']}: {sum(advance['conditions'].values())}/{len(advance['conditions'])} conditions.**", '',
        '![Every fit and its measured cost](benchmark.png)', '',
        ('All three arms use the same public-prefix-only stage until elapsed 10 seconds, then joint H1/H2 '
         'supervision until elapsed 40 seconds from one start. They share 352 stored parameters and fresh data. '
         'Late updates restore model, optimizer and cursor state; all attempted work remains charged.'), '',
        '[Protocol](../finite-rounded-learning-protocol.md) · [All saved numbers](summary.json) · [Original evidence receipt](receipt.json)', '',
        '| Arm | Transition | Initial comparison |', '|---|---|---|',
        '| Original free | Column softmax | Original seeded logits |',
        '| Matched free | Column softmax | Effective transition matched to rounded initialization within 1e-12 |',
        '| Rounded | Four fixed sweeps, slack contractions and rank-one correction | Same original raw logits |', '',
        ('Emission, hazard and cost-head initial parameters are paired. The rounded constraint has 196 ideal '
         'transition degrees versus 224 for free columns, despite equal raw parameter counts. Function matching '
         "does not match gradients or update counts. The construction uses the task's doubly stochastic structure "
         'and still permits uniform mixing.'), '',
        '| Criterion | Original free | Matched free | Rounded |', '|---|---|---|---|']
    for criterion in CRITERIA:
        cells = []
        for arm in ARMS:
            gate = n['gates'][arm][criterion]
            cells.append(f"{'PASS' if gate['passed'] else 'FAIL'} {sum(gate['conditions'].values())}/{len(gate['conditions'])}")
        lines.append('| ' + criterion + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', ('Advance requires all three original rounded-arm criteria. Against each control, all six '
        'paired H4/H8 regret differences must be nonpositive, both horizon means must improve at least 10% '
        'from positive control means, and mean full fit time must be at most 1.05 times the control. '
        'No mean or control substitution rescues a failed condition.'), '',
        '| Advance condition | Outcome |', '|---|---|']
    lines += [f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in advance['conditions'].items()]
    lines += ['', '## Paired long-horizon comparisons', '',
        '| Control | Seed | H | Rounded regret | Control regret | Difference | Relative reduction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in n['allocation_comparisons']['paired']:
        value = row['blind_regret']
        reduction = 'undefined (zero control)' if value['control'] == 0 else f"{100 * (1 - value['candidate'] / value['control']):.3f}%"
        lines.append(f"| {row['control']} | {row['seed']} | {row['horizon']} | {value['candidate']:.7g} | "
                     f"{value['control']:.7g} | {value['difference']:.7g} | {reduction} |")
    lines += ['', '## Three-seed means', '',
        '| Arm | H | Blind MSE | Regret | Observed KL |', '|---|---:|---:|---:|---:|']
    for row in n['three_seed_means']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_cost_mse']:.7g} | {row['blind_regret']:.7g} | {row['observed_kl']:.7g} |")
    lines += ['', n['mean_scope'], '', '## Every training allocation', '',
        '| Arm | Seed | Accepted joint | Accepted prefix | All attempted | Timed s | Overrun s | Summary s | Construction s | Full fit s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in n['training_costs']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['accepted_joint']} | {row['accepted_prefix']} | {row['attempted']} | "
                     f"{row['timed_seconds']:.6f} | {row['overrun_seconds']:.6f} | {row['summary_seconds']:.6f} | "
                     f"{row['construction_seconds']:.6f} | {row['seconds']:.6f} |")
    lines += ['', n['timing_scope'], '', ('Timing columns are nested, not additive. All arms freeze the cost head '
        'in prefix training, then start joint training at batch zero with fresh Adam state. Identical eligibility '
        'windows are not equal FLOPs, accepted updates or actual wall time. Full construction and structural '
        'forward-operation counters are retained in the summary; they do not count backward FLOPs.'), '',
        '## Correction at every rounded boundary', '',
        '| Seed | Boundary | Pre-round row residual | Pre-round column residual | Largest action correction mass | Largest correction entry | Largest change | Final row residual | Final column residual |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in n['rounded_boundary_diagnostics']:
        diagnostic = row['rounding']
        values = (diagnostic['pre_round_row_residual_max'], diagnostic['pre_round_column_residual_max'],
                  max(diagnostic['correction_mass']), diagnostic['correction_maximum_entry'],
                  diagnostic['pre_to_final_maximum_change'], diagnostic['row_residual_max'],
                  diagnostic['column_residual_max'])
        lines.append(f"| {row['seed']} | {row['boundary']} | " + ' | '.join(f'{v:.7g}' for v in values) + ' |')
    lines += ['', ("These nine records are independently reconstructed from the rounded arm's saved initial, "
        'stage-one boundary and final parameters. All 27 arm/boundary records remain in the summary; the 18 free '
        'records use their actual column softmax and have no correction values. Each correction uses four sweeps '
        'and fixed slack 1e-8, with final residual tolerance 1e-12. Boundary values do not bound unsaved intermediate '
        'training states or replay their gradients. A balanced input X becomes (1-1e-8)X+1e-8 U, so the correction '
        'is an intentional parameterization, not a numerical-only repair.'), '',
        '## Every endpoint metric', '',
        '| Arm | Seed | H | Cases | Blind MSE | Regret | Survival MAE | Observed MSE | Observed survival MAE | Observed KL | Shuffled regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in n['rows']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['horizon']} | {row['cases']} | "
                     + ' | '.join(f'{row[key]:.7g}' for key in METRICS) + ' |')
    lines += ['', '## Evidence and limits', '',
        (f"The original engineering log records {n['engineering_tests_passed']} passing tests and "
         f"{n['engineering_test_warnings']} warning. Qualification includes a complete independent audit of its "
         'small saved run. All nine scientific final checkpoints precede fresh DEV generation. The independent '
         'audit makes 45 array reads, including 27 parameter checkpoints, plus 27 optimizer JSON reads; it '
         'reconstructs public-history targets, metrics, transition boundaries and work records without model, '
         'optimizer or environment calls.'), '', '| Original supervised phase | Seconds |', '|---|---:|']
    lines += [f'| {key} | {value:.3f} |' for key, value in n['phase_seconds'].items()]
    lines += [f"| Total of these three original phases | {n['total_supervised_seconds']:.3f} |"]
    lines += ['', ('All seeds and registered conditions are retained. Historical update losses, timings and '
        'intermediate hash records are source-qualified attestations, not a training replay. The exact reference '
        'uses known dynamics; learned arms receive public histories. Observed filtering has intervening observations '
        'and is distinct from blind extrapolation. These synthetic results and their privileged initial readout '
        'do not establish statistical significance, latent identification, native transfer, biological wiring '
        'advantages or architectural novelty. The earlier failed balanced integration remains closed.'), '']
    return '\n'.join(lines)


def render(study, output, registration_sha256):
    require(registration_sha256 == REGISTRATION_SHA256, 'explicit registered scientific identity')
    output = Path(output)
    overview_path = ROOT / 'research/finite-rounded-learning-results.md'
    require(output == ROOT / 'research/finite-rounded-learning-results' and not output.exists()
            and not output.is_symlink() and output.parent.resolve() == output.parent
            and not overview_path.exists() and not overview_path.is_symlink(),
            'exclusive exact report directory')
    auth = authenticate(Path(study))
    numbers = extract(auth)
    output.mkdir(exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(numbers, indent=2, sort_keys=True, allow_nan=False) + '\n')
    report = document(numbers)
    (output / 'report.md').write_text(report)
    figure(numbers, output / 'benchmark.png')
    require(all(descriptor(Path(p)) == pin for p, pin in auth['inputs'].items()), 'original evidence unchanged')
    for phase in auth['phases'].values():
        require(inventory(phase['directory']) == phase['receipt']['files'], 'complete phase unchanged')
    proof = auth['plan']['kernel_qualification']
    require(inventory(Path(proof['folder'])) == proof['files'], 'numerical prerequisite unchanged')
    require(inventory(Path(proof['publication'])) == proof['publication_files'], 'primitive publication unchanged')
    overview = report.replace('(../finite-rounded-learning-protocol.md)', '(finite-rounded-learning-protocol.md)')
    for name in ('benchmark.png', 'summary.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(finite-rounded-learning-results/' + name + ')')
    with overview_path.open('x') as stream:
        stream.write(overview)
    receipt = {'version': VERSION, 'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'renderer': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'overview': {'path': str(overview_path), **descriptor(overview_path)},
        'inputs': auth['inputs'], 'sources': auth['plan']['sources'], 'kernel_qualification': proof,
        'files': {name: descriptor(output / name) for name in ('summary.json', 'report.md', 'benchmark.png')},
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls',
                                'optimizer_calls', 'native_calls', 'teacher_calls'), 0),
        'result_reads_after_original_closure': True, 'model_selection': False, 'visual_review_required': True}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    result = render(args.study, args.output, args.registration_sha256)
    print(json.dumps({'version': result['version'], 'files': result['files']}, sort_keys=True))


if __name__ == '__main__':
    main()
