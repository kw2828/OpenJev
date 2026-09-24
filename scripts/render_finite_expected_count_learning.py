"""Publish only closed, authenticated JSON for expected-count prefix learning.

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
VERSION = 'finite-expected-count-learning-report-v1'
STUDY_NAME = 'finite-expected-count-learning-v1'
REGISTRATION_SHA256 = 'cbf99064bff1e22e7f6502ce78c796d2ac70735d7c2f08406215b1bf14b00597'
ARMS = ('joint_only', 'gradient_prefix', 'em_prefix')
LABELS = ('Joint only', 'Gradient prefix', 'EM prefix')
SEEDS = (429261001, 429261002, 429261003)
HORIZONS = (1, 2, 4, 8)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')
CONFIG = {'seed_namespace': 429260924, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2,
          'dev_horizon': 8, 'pretraining_seconds': 10.}


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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_expected_count_learning_worker.py'
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
             'training-orders.jsonl', 'training-epochs.jsonl', 'checkpoint-barrier.json',
             'prediction-times.jsonl', 'oracle-train-check.json', 'oracle-base-check.json',
             'train-prefix.npz', 'base-prefix.npz', 'pretraining.jsonl'}
    for arm in ARMS:
        for seed in SEEDS:
            names |= {f'{arm}-{seed}.npz', f'prefix-{arm}-{seed}-base.npz',
                      f'pretraining-initial-{arm}-{seed}.npz', f'pretraining-final-{arm}-{seed}.npz'}
    require(len(names) == 54, 'complete fixed producer inventory')
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
            and len(plan['sources']) == 36, 'fixed study, configuration and source roster')
    for name, expected in plan['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(ROOT / name) == expected, 'unchanged registered source: ' + name)
    # This pinned helper imports only standard-library metadata/clock support.
    import finite_expected_count_learning_worker as worker

    proof = plan['kernel_qualification']
    require(Path(proof['folder']) == ROOT / 'output/finite-expected-count-qualification-v1'
            and inventory(Path(proof['folder'])) == proof['files'], 'complete unchanged numerical prerequisite')
    kernel_receipt_path = Path(proof['folder']) / 'run-01.receipt.json'
    kernel = read(kernel_receipt_path)
    require(kernel['status'] == 'COMPLETED' and kernel['engineering_gate'] == 'PASS'
            and kernel['launch'] == read(Path(kernel['supervision']))
            and inventory(Path(kernel['output'])) == kernel['files'], 'original successful numerical prerequisite')
    worker.closed_producer(kernel)
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
                                           '-p', 'no:cacheprovider', *plan['tests']], 'exact source-qualified engineering commands')
    # No fit or audit result JSON is opened before all joins above succeed.
    saved = read(fit['directory'] / 'summary.json')
    report = read(audit['directory'] / 'audit.json')
    require(saved['version'] == STUDY_NAME and saved['config'] == CONFIG
            and report['version'] == 'finite-expected-count-learning-audit-v1'
            and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == report['gates'], 'authoritative independent saved-output audit')
    require(set(saved['files']) == producer_files()
            and fit['receipt']['files'] == {**saved['files'], 'summary.json': descriptor(fit['directory'] / 'summary.json')}
            and fit['receipt']['result'] == {'fits': 9, 'rows': 36}
            and report['counts'] == {'array_decodes': 36, 'checkpoint_decodes': 18, 'model_calls': 0,
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
    return {'plan': plan, 'plan_path': plan_path, 'phases': phases, 'summary': saved,
            'audit': report, 'inputs': inputs}


def extract(auth):
    saved, report, plan = auth['summary'], auth['audit'], auth['plan']
    rows, baselines = report['rows'], report['baseline_rows']
    pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in rows}
    require(len(rows) == len(keyed) == 36 and set(keyed) == {(a, s, h) for a, s in pairs for h in HORIZONS}
            and len(baselines) == 4 and {r['horizon'] for r in baselines} == set(HORIZONS), 'all endpoint cells and uniform references')
    for row in rows:
        require(row['regime'] == 'base' and row['cases'] == report['data_cases']['base']
                and all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in METRICS), 'finite complete endpoint row')
    require(set(report['gates']) == set(ARMS), 'all three arms retained')
    for arm in ARMS:
        require(set(report['gates'][arm]) == set(CRITERIA), 'all unchanged criteria retained')
        for criterion in CRITERIA:
            gate = report['gates'][arm][criterion]
            require(type(gate['passed']) is bool and gate['conditions']
                    and all(type(v) is bool for v in gate['conditions'].values())
                    and gate['passed'] is all(gate['conditions'].values())
                    and gate['status'] == criterion + ('_PASS' if gate['passed'] else '_FAIL'), 'unaltered criterion outcome')
    for name in ('fits', 'pretraining', 'prediction_times', 'prefix_prediction_times'):
        require(len(saved[name]) == 9 and {(r['arm'], r['seed']) for r in saved[name]} == pairs, 'complete ' + name)
    require(len(report['prefix_rows']) == 9 and {(r['arm'], r['seed']) for r in report['prefix_rows']} == pairs,
            'all-attempt prefix diagnostics retained')
    barrier = saved['checkpoint_barrier']
    require(barrier['fit_count'] == saved['counts']['fit_count'] == 9 and barrier['dev_generation_count'] == 0
            and barrier['oracle_train_verified'] is True and saved['counts']['pretraining_runs'] == 9
            and saved['counts']['pretraining_checkpoint_writes'] == 18, 'all original stages before DEV')
    require(saved['structural_work'] == report['structural_work']
            and set(saved['structural_work']) == set(ARMS), 'independently checked structural work')
    stages = {(r['arm'], r['seed']): r for r in saved['pretraining']}
    fits = {(r['arm'], r['seed']): r for r in saved['fits']}
    costs = []
    for arm in ARMS:
        for seed in SEEDS:
            fit, stage = fits[arm, seed], stages[arm, seed]
            require(fit['parameter_metadata']['parameter_count'] == 352
                    and fit['parameter_metadata']['parameter_bytes'] == 2816
                    and fit['initial_state_sha256'] == stage['final_state_sha256'], 'common model and retained-prefix to joint-fit boundary')
            result = stage['result']
            values = (fit['construction_seconds'], stage['seconds'], fit['seconds'])
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), 'finite nonoverlapping training components')
            costs.append({'arm': arm, 'seed': seed, 'construction_seconds': values[0],
                'pretraining_seconds': values[1], 'joint_fit_seconds': values[2], 'total_training_seconds': math.fsum(values),
                'accepted_updates': result['accepted_updates'], 'attempted_updates': result['attempted_updates'],
                'timed_seconds': result['timed_seconds'], 'overrun_seconds': result['overrun_seconds'],
                'final_summary_seconds': result['final_summary_seconds'], 'joint_updates': fit['updates']})
    for row in saved['prediction_times']:
        require(row['oracle_prefix_input'] is False and row['model_state_before'] == row['model_state_after']
                == fits[row['arm'], row['seed']]['final_state_sha256'], 'unchanged learner evaluated without oracle input')
    means = [{'arm': arm, 'horizon': h, **{key: math.fsum(keyed[arm, s, h][key] for s in SEEDS) / 3 for key in METRICS}}
             for arm in ARMS for h in HORIZONS]
    contrasts = report['dynamics_comparisons']['paired']
    require(len(contrasts) == 12 and {(r['control'], r['seed'], r['horizon']) for r in contrasts}
            == {(a, s, h) for a in ('joint_only', 'gradient_prefix') for s in SEEDS for h in (4, 8)}, 'every audited EM/control pair')
    log_path = auth['phases']['qualify']['directory'] / 'command-1.log'
    log = regular(log_path).read_text()
    count_matches = re.findall(r'(?m)^\s*(\d+) passed(?:, [^\n]+)? in [^\n]+$', log)
    require(len(count_matches) == 1, 'one unambiguous authenticated pytest completion summary')
    phase_seconds = {key: value['terminal']['wall_seconds'] for key, value in auth['phases'].items()}
    return {'version': VERSION, 'config': plan['config'], 'gates': report['gates'], 'rows': rows,
        'baseline_rows': baselines, 'three_seed_means': means, 'paired_em_vs_controls_h4_h8': contrasts,
        'training_costs': costs, 'fits': saved['fits'], 'pretraining': saved['pretraining'],
        'pretraining_audit': report['pretraining'], 'prefix_rows': report['prefix_rows'],
        'prediction_times': saved['prediction_times'], 'prefix_prediction_times': saved['prefix_prediction_times'],
        'dataset_counts': saved['dataset_counts'], 'data_cases': report['data_cases'], 'counts': saved['counts'],
        'counts_scope': saved['counts_scope'], 'audit_counts': report['counts'],
        'structural_work': saved['structural_work'], 'structural_work_scope': saved['structural_work_scope'],
        'generation_seconds': saved['generation_seconds'], 'generation_timing_scope': saved['generation_timing_scope'],
        'phase_seconds': phase_seconds, 'successful_phase_seconds': math.fsum(phase_seconds.values()),
        'engineering_tests_passed': int(count_matches[0]), 'engineering_test_log': {'path': str(log_path), **descriptor(log_path)},
        'sources': plan['sources'], 'limitations': report['limitations'],
        'mean_scope': 'Means of three separately fitted policies on common cases; not an ensemble or significance test.',
        'timing_scope': 'Construction + full pretraining wrapper + joint fit. Timed eligibility, overshoot and final summary are nested within the wrapper, not added twice.',
        'compute_matched': False, 'architecture_claim': False, 'latent_identification_claim': False,
        'native_environment_claim': False, 'novelty_claim': False, 'model_selection': False}


def figure(numbers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors, markers = ('#666666', '#0072B2', '#D55E00'), ('o', 's', '^')
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in numbers['rows']}
    costs = {(r['arm'], r['seed']): r for r in numbers['training_costs']}
    baseline = {r['horizon']: r['blind_regret'] for r in numbers['baseline_rows']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for arm, color in zip(ARMS, colors, strict=True):
        for seed, marker in zip(SEEDS, markers, strict=True):
            axes[0].plot(HORIZONS, [keyed[arm, seed, h]['blind_regret'] for h in HORIZONS],
                         color=color, marker=marker, linewidth=.8, markersize=4, alpha=.4)
        axes[0].plot(HORIZONS, [math.fsum(keyed[arm, s, h]['blind_regret'] for s in SEEDS) / 3 for h in HORIZONS],
                     color=color, linewidth=2.2)
    axes[0].plot(HORIZONS, [baseline[h] for h in HORIZONS], color='#111111', linestyle='--')
    axes[0].axvspan(.8, 2.2, color='#DDE4ED', alpha=.3, zorder=0)
    axes[0].set(title='Blind decision regret: every seed and arm', xlabel='Forecast horizon', ylabel='Expected cost regret')
    axes[0].set_xticks(HORIZONS)
    for seed, marker in zip(SEEDS, markers, strict=True):
        values = [costs[a, seed]['total_training_seconds'] for a in ARMS]
        axes[1].plot(range(3), values, color='#888888', linewidth=.7, alpha=.45)
        for i, color in enumerate(colors):
            axes[1].scatter(i, values[i], color=color, marker=marker, s=40)
    for i, arm in enumerate(ARMS):
        mean = math.fsum(costs[arm, s]['total_training_seconds'] for s in SEEDS) / 3
        axes[1].plot((i - .12, i + .12), (mean, mean), color=colors[i], linewidth=3)
    axes[1].set_xticks(range(3), ('Joint only', 'Gradient\nprefix', 'EM prefix'))
    axes[1].set(xlim=(-.4, 2.4), title='Full measured training cost per fit', ylabel='Seconds: construction + prefix stage + joint fit')
    for axis in axes:
        axis.grid(axis='y', alpha=.18)
        axis.set_ylim(bottom=0)
    handles = [Line2D([], [], color=c, linewidth=2, label=label) for c, label in zip(colors, LABELS, strict=True)]
    handles += [Line2D([], [], color='#111111', linestyle='--', label='Known-dynamics uniform reference')]
    handles += [Line2D([], [], color='#777777', marker=m, linewidth=.7, label=f'Seed {s}') for s, m in zip(SEEDS, markers, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Synthetic prefix-learning diagnostic: same model, different pretraining', fontsize=14, fontweight='bold')
    fig.text(.5, .14, 'All nine fits shown. Thick marks are arithmetic means, not an ensemble. Shading: trained horizons H1/H2.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.25, wspace=.28)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(n):
    lines = ['# Expected-count prefix learning: synthetic diagnostic', '',
        ('All nine fits use the same 352-parameter recurrent model and paired original parameters. '
        'Only the public-prefix pretraining method changes; every arm then receives the same 480-epoch joint-training recipe.'), '',
        '![Every fit seed, blind regret and full measured training cost](benchmark.png)', '',
        ('[Frozen protocol](../finite-expected-count-learning-protocol.md) · [Full saved report](summary.json) · '
        '[Original closure and source receipt](receipt.json). Original successful process closures and all source and payload hashes '
        'were checked before reading result metrics.'), '',
        '## All unchanged criteria', '',
        '| Criterion | Joint only | Gradient prefix | EM prefix |', '|---|---|---|---|']
    for criterion in CRITERIA:
        cells = []
        for arm in ARMS:
            gate = n['gates'][arm][criterion]
            cells.append(f"{'PASS' if gate['passed'] else 'FAIL'} ({sum(gate['conditions'].values())}/{len(gate['conditions'])})")
        lines.append('| ' + criterion + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', ('Every required condition must pass for every seed. SHORT uses H1/H2 cost MSE and regret at most half '
        'the positive uniform reference and observed KL at most 0.1 nats. BLIND uses H4/H8 cost thresholds and H8 survival '
        'MAE at most 0.05. OBSERVED uses H4/H8 KL at most 0.1. Required endpoint support is 256 TRAIN and 64 DEV. '
        'Favorable means or prefix likelihood cannot rescue failures; the full condition dictionaries remain in the summary.'), '',
        '## Longer-horizon means', '',
        '| Arm | H | Blind regret | Blind MSE | Observed MSE | Observed KL |', '|---|---:|---:|---:|---:|---:|']
    for r in n['three_seed_means']:
        if r['horizon'] in (4, 8):
            lines.append(f"| {r['arm']} | {r['horizon']} | {r['blind_regret']:.6g} | {r['blind_cost_mse']:.6g} | {r['observed_cost_mse']:.6g} | {r['observed_kl']:.6g} |")
    lines += ['', n['mean_scope'], '', '## Every paired EM comparison', '',
        ('Signed differences are EM minus control for the same seed and cases. Negative favors EM for that metric. '
        'These are descriptive differences, with no significance or additional continuation criterion.'), '',
        '| Control | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in n['paired_em_vs_controls_h4_h8']:
        values = ' | '.join(f"{r['metrics'][key]['difference']:+.6g}" for key in ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl'))
        lines.append(f"| {r['control']} | {r['seed']} | {r['horizon']} | {values} |")
    lines += ['', '## Complete training costs and deadline outcomes', '',
        '| Arm | Seed | Accepted / attempted | Timed stage s | Charged overrun s | Outside-stage summary s | Full prefix wrapper s | Construction s | Joint fit s | Total s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in n['training_costs']:
        values = ' | '.join(f'{r[key]:.6f}' for key in ('timed_seconds', 'overrun_seconds', 'final_summary_seconds',
                            'pretraining_seconds', 'construction_seconds', 'joint_fit_seconds', 'total_training_seconds'))
        lines.append(f"| {r['arm']} | {r['seed']} | {r['accepted_updates']} / {r['attempted_updates']} | {values} |")
    lines += ['', ('The two active prefix methods receive 10 seconds of update eligibility, not identical total wall time or FLOPs. '
        'A complete update must finish within the window. Late work is rolled back, retained in the trace and charged; '
        'at least one accepted update is required. Full prefix-wrapper time includes boundary snapshots and final diagnostics. '
        'The first three timing columns are nested within that wrapper and are not added again to total training time.'), '',
        ('The eligibility clock is monotonic `perf_counter`; the outer phase deadline is suspend-inclusive. Machine contention '
        'can change update counts. EM uses expected-count MAP updates; gradient pretraining uses persistent Adam. Both optimize '
        'the same likelihood plus 0.001 log-prior, with no cost-head updates. Joint training starts from the retained prefix state, '
        'uses a fresh Adam optimizer, and does not add the pretraining prior.'), '',
        '| Arm | Seed | Joint updates | Gradient passes | Expected-count passes | Prefix event NLL (DEV) |',
        '|---|---:|---:|---:|---:|---:|']
    stages = {(r['arm'], r['seed']): r for r in n['pretraining']}
    prefix = {(r['arm'], r['seed']): r for r in n['prefix_rows']}
    for r in n['training_costs']:
        key = r['arm'], r['seed']
        work = stages[key]['result'].get('work', {})
        lines.append(f"| {r['arm']} | {r['seed']} | {r['joint_updates']} | {work.get('full_gradient_passes', 0)} | "
                     f"{work.get('numpy_expectation_passes', 0)} | {prefix[key]['mean_nll']:.6g} |")
    lines += ['', ('Complete attempted-update traces, accepted and restored hashes, probability checks, diagnostic passes, '
        'event exposures, copies, hashes and rollback work remain in `pretraining` in the summary. '
        'The 21 common-training/evaluation structural-work routes are also retained; they are not exhaustive FLOP measurements.'), '',
        '## Data and whole-phase costs', '',
        '| Split | Attempted prefixes | Endpoint eligible | Found-terminated | Valid prefix events |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'base'):
        r = n['dataset_counts'][split]
        lines.append(f"| {split} | {r['attempted']} | {r['retained']} | {r['discarded_found']} | {r['valid_prefix_events']} |")
    lines += ['', (f"The authenticated engineering log reports {n['engineering_tests_passed']} passing tests. "
        'All nine final joint checkpoints were saved before DEV generation. Prefix likelihood includes every attempted history '
        'and its first found event, without post-terminal padding or replacement cases.'), '',
        '| Original closed phase | Seconds including launch and cleanup |', '|---|---:|']
    for phase, seconds in n['phase_seconds'].items():
        lines.append(f'| {phase} | {seconds:.3f} |')
    lines += [f"| Sum of successful phases | {n['successful_phase_seconds']:.3f} |", '',
        ('Whole-phase durations already contain their nested operations. The producer includes generation, exact-control checks, '
        'all training, evaluation and writes. Separate generation and inference timings remain in the summary. '
        'These are aggregate workload timings, not single-decision latency.'), '',
        '## Interpretation limits', '',
        ('This tests a learning recipe on a fixed synthetic environment, not a new architecture. All learners receive public '
        'tokens only, but share a privileged world-aligned initial cost head and known uniform reset prior. The exact correctness '
        'control alone receives the true boundary posterior; the uniform reference knows dynamics while discarding history. '
        'Observed filtering consumes intervening labels and is distinct from blind extrapolation.'), '',
        ('The audit reconstructs targets and metrics and decodes 18 prefix-boundary checkpoints for hash/head/retained-state checks. '
        'It does not replay optimizer updates or independently establish historical timings; those remain source-qualified attestations. '
        'Joint-final checkpoints remain opaque. No latent-state identification, calibration, scenario-shift robustness, robotics or '
        'native transfer, novelty, or ICLR-readiness claim follows from this diagnostic. Earlier failed studies remain closed.'), '']
    lines += ['## Appendix: every endpoint result', '',
        'All 36 audited arm/seed/horizon cells are shown, without filtering or seed selection.', '',
        '| Arm | Seed | H | Cases | Blind regret | Blind cost MSE | Blind survival MAE | Observed cost MSE | Observed survival MAE | Observed KL | Shuffled blind regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in n['rows']}
    for arm in ARMS:
        for seed in SEEDS:
            for horizon in HORIZONS:
                row = keyed[arm, seed, horizon]
                values = ' | '.join(f'{row[key]:.9g}' for key in (
                    'blind_regret', 'blind_cost_mse', 'blind_survival_mae', 'observed_cost_mse',
                    'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret'))
                lines.append(f"| {arm} | {seed} | {horizon} | {row['cases']} | {values} |")
    lines.append('')
    return '\n'.join(lines)


def render(studyfolder, outputfolder, registration_sha256):
    require(registration_sha256 == REGISTRATION_SHA256, 'registered scientific identity must be explicit')
    outputfolder = Path(outputfolder)
    require(outputfolder == ROOT / 'research/finite-expected-count-learning-results'
            and not outputfolder.exists(), 'exclusive exact publication directory')
    auth = authenticate(Path(studyfolder))
    numbers = extract(auth)
    outputfolder.mkdir(exist_ok=False)
    (outputfolder / 'summary.json').write_text(json.dumps(numbers, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (outputfolder / 'report.md').write_text(document(numbers))
    figure(numbers, outputfolder / 'benchmark.png')
    for name, expected in auth['inputs'].items():
        require(descriptor(Path(name)) == expected, 'publication leaves original input bytes unchanged')
    for phase in auth['phases'].values():
        require(inventory(phase['directory']) == phase['receipt']['files'], 'unchanged complete phase inventory after render')
    proof = auth['plan']['kernel_qualification']
    require(inventory(Path(proof['folder'])) == proof['files'], 'unchanged complete numerical prerequisite after render')
    receipt = {'version': VERSION, 'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'renderer': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'inputs': auth['inputs'], 'sources': auth['plan']['sources'], 'kernel_qualification': proof,
        'files': {name: descriptor(outputfolder / name) for name in ('summary.json', 'report.md', 'benchmark.png')},
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls',
                                'optimizer_calls', 'native_calls', 'teacher_calls'), 0),
        'result_reads_after_original_closure': True, 'model_selection': False, 'visual_review_required': True}
    (outputfolder / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.study, args.output, args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
