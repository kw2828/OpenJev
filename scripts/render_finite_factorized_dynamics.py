"""Render only authenticated, closed JSON from the factorized-dynamics diagnostic.

This publication helper performs no array/checkpoint decode, model construction,
generation or optimization. All original sources, closures and payload bytes
are checked before reading result metrics. No result selection is supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'finite-factorized-dynamics-report-v1'
STUDY_NAME = 'finite-factorized-dynamics-v1'
ARMS = ('factorized', 'matched_free', 'dense_free')
SEEDS = (426261001, 426261002, 426261003)
CONFIG = {'seed_namespace': 426260924, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8}
HORIZONS = (1, 2, 4, 8)
ROUTES = ('training_blind', 'training_observed', 'training_prefix', 'evaluation_blind', 'evaluation_observed', 'evaluation_shuffled', 'evaluation_prefix')
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Factorized', 'Function-matched free', 'Dense free')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')


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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_factorized_dynamics_worker.py'
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


def authenticate(studyfolder):
    require(studyfolder == ROOT / 'output' / STUDY_NAME, 'exact factorized-dynamics study directory')
    plan_path = regular(studyfolder / 'study-registration.json')
    plan = read(plan_path)
    require(plan['version'] == STUDY_NAME and plan['root'] == str(ROOT)
            and plan['mode'] == 'study' and plan['config'] == CONFIG, 'registered factorized-dynamics study')
    for name, expected in plan['sources'].items():
        path = ROOT / name
        require(path.is_relative_to(ROOT) and '..' not in Path(name).parts and descriptor(path) == expected,
                'current registered source: ' + name)
    fit = closed_phase(plan, plan_path, 'fit')
    audit = closed_phase(plan, plan_path, 'audit')
    require(fit['terminal']['finished_ns'] <= audit['terminal']['started_ns']
            and audit['receipt']['producer_receipt'] == descriptor(fit['receipt_path'])
            and audit['receipt']['producer_terminal'] == descriptor(fit['terminal_path']), 'original fit closure precedes audit')
    q = plan['qualification']
    q_receipt_path = regular(Path(q['path']))
    require(descriptor(q_receipt_path) == q['descriptor'], 'registered qualification receipt bytes')
    q_receipt = read(q_receipt_path)
    q_plan_path = regular(Path(q_receipt['plan']))
    q_plan = read(q_plan_path)
    require(q_plan['version'] == STUDY_NAME and q_plan['mode'] == 'engineering' and q_plan['root'] == str(ROOT)
            and q_plan['config'] == CONFIG
            and q_plan['sources'] == plan['sources'], 'same qualified source closure')
    qualify = closed_phase(q_plan, q_plan_path, 'qualify')
    require(qualify['receipt_path'] == q_receipt_path and descriptor(qualify['terminal_path']) == q['terminal']
            and qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns'], 'qualification closed before original fit')
    # First result reads happen only after successful original closure and pins.
    saved = read(fit['directory'] / 'summary.json')
    report = read(audit['directory'] / 'audit.json')
    require(saved['version'] == 'finite-factorized-dynamics-v1' and report['version'] == 'finite-factorized-dynamics-audit-v1'
            and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == report['gates'], 'independent audit and explicit external closure boundary')
    require(saved['config'] == plan['config'] and tuple(saved['config']['fit_seeds']) == SEEDS
            and saved['config']['epochs'] == 480 and saved['counts']['fit_count'] == 9
            and saved['checkpoint_barrier']['fit_count'] == 9
            and saved['checkpoint_barrier']['dev_generation_count'] == 0
            and saved['checkpoint_barrier']['oracle_train_verified'] is True, 'complete qualified oracle and fit barrier')
    pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    expected_files = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
        'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl',
        'training-orders.jsonl', 'training-epochs.jsonl', 'checkpoint-barrier.json', 'prediction-times.jsonl',
        'oracle-train-check.json', 'oracle-base-check.json', 'train-prefix.npz', 'base-prefix.npz'}
    expected_files |= {f'{arm}-{seed}.npz' for arm, seed in pairs}
    expected_files |= {f'prefix-{arm}-{seed}-base.npz' for arm, seed in pairs}
    require(len(expected_files) == 35 and set(saved['files']) == expected_files
            and fit['receipt']['files'] == {**saved['files'], 'summary.json': descriptor(fit['directory'] / 'summary.json')}
            and set(audit['receipt']['files']) == {'audit.json'}, 'exact original fit and audit artifact rosters')
    require(fit['receipt']['result'] == {'fits': 9, 'rows': 36}
            and saved['counts']['readout_snapshot_evaluations'] == 18
            and saved['counts']['readout_snapshot_softmax_evaluations'] == 18
            and saved['counts']['dynamics_snapshot_evaluations'] == 18
            and report['counts'] == {'array_decodes': 18, 'checkpoint_decodes': 0, 'model_calls': 0,
                'optimizer_calls': 0, 'world_or_generator_calls': 0, 'native_calls': 0}
            and 'continuation' not in report and 'continuation' not in audit['receipt'],
            'exact nine-fit and independent audit work, without the prior replication continuation rule')
    finite_tree(saved)
    finite_tree(report)
    return plan_path, plan, {'qualify': qualify, 'fit': fit, 'audit': audit}, saved, report


def extract(plan, phases, saved, report):
    rows, baselines = report['rows'], report['baseline_rows']
    expected = {(arm, seed, 'base', h) for arm in ARMS for seed in SEEDS for h in HORIZONS}
    require(len(rows) == 36 and {(r['arm'], r['seed'], r['regime'], r['horizon']) for r in rows} == expected,
            'all three learned arms, three seeds and four horizons')
    require(len(baselines) == 4 and {(r['regime'], r['horizon']) for r in baselines} == {('base', h) for h in HORIZONS},
            'complete uniform-state reference')
    require(len(saved['fits']) == 9 and {(r['arm'], r['seed']) for r in saved['fits']}
            == {(arm, seed) for arm in ARMS for seed in SEEDS}, 'all nine fit records')
    require(set(report['data_cases']) == {'train', 'base'}, 'exact TRAIN and base DEV support')
    for split, count in report['data_cases'].items():
        require(type(count) is int and count > 0 and count == saved['dataset_counts'][split]['retained'], 'audited retained cases')
    for row in rows:
        require(row['cases'] == report['data_cases']['base']
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in METRICS), 'finite retained model cell')
    require(set(report['gates']) == set(ARMS), 'all three diagnostic arms')
    for arm in ARMS:
        require(set(report['gates'][arm]) == set(CRITERIA), 'three distinct diagnostic criteria')
        for criterion in CRITERIA:
            result = report['gates'][arm][criterion]
            require(type(result['passed']) is bool and all(type(value) is bool for value in result['conditions'].values())
                    and result['passed'] is all(result['conditions'].values())
                    and result['status'] == criterion + ('_PASS' if result['passed'] else '_FAIL'), 'unaltered complete criterion')
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    means = [{'arm': arm, 'regime': 'base', 'horizon': h, 'fit_seeds': list(SEEDS),
              **{metric: math.fsum(keyed[arm, seed, h][metric] for seed in SEEDS) / 3 for metric in METRICS}}
             for arm in ARMS for h in (4, 8)]
    paired = [{'candidate': 'factorized', 'control': control, 'seed': seed,
               'regime': 'base', 'horizon': h,
               **{metric: keyed['factorized', seed, h][metric] - keyed[control, seed, h][metric]
                  for metric in ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl')}}
              for control in ('matched_free', 'dense_free') for h in (4, 8) for seed in SEEDS]
    fit_map = {(row['arm'], row['seed']): row for row in saved['fits']}
    comparisons = report['dynamics_comparisons']
    require(len(comparisons['initial_function_checks']) == 3
            and {row['seed'] for row in comparisons['initial_function_checks']} == set(SEEDS),
            'three independently checked initial-function pairs')
    for row in comparisons['initial_function_checks']:
        require(row['passed'] is True and set(row['maximum_absolute_differences'])
                == {'reset_emission', 'observed', 'found', 'blind'}
                and all(0 <= value <= 1e-12 for value in row['maximum_absolute_differences'].values())
                and 0 <= row['head_maximum_absolute_difference'] <= 1e-12,
                'matched initial functions and common learned head, not parameter equality')
    require(len(comparisons['fits']) == 9
            and {(row['arm'], row['seed']) for row in comparisons['fits']} == set(fit_map)
            and len(comparisons['paired']) == 12 and len(comparisons['means']) == 12,
            'complete independently audited dynamics comparisons')
    for row in saved['fits']:
        parameter_count = 352 if row['arm'] == 'factorized' else 1120
        require(row['parameter_metadata']['parameter_count'] == parameter_count
                and row['parameter_metadata']['parameter_bytes'] == 8 * parameter_count
                and row['parameter_metadata']['buffer_bytes'] == 0
                and row['prefix_loss_weight'] == 1, 'declared dynamics capacity and common auxiliary coefficient')
        require(type(row['construction_seconds']) in (int, float) and row['construction_seconds'] >= 0,
                'separately measured construction time')
        for name in ('dynamics_initial', 'dynamics_final'):
            require(set(row[name]) == {'reset_emission', 'observed', 'found', 'blind', 'work'}
                    and all(type(value) is int and value >= 0 for value in row[name]['work'].values()),
                    'complete saved dynamics snapshots and separate work')
    prefix_rows = report['prefix_rows']
    require(len(prefix_rows) == 9 and {(r['arm'], r['seed']) for r in prefix_rows} == set(fit_map),
            'complete all-attempt prefix diagnostics')
    require(len(saved['prefix_prediction_times']) == 9
            and {(r['arm'], r['seed']) for r in saved['prefix_prediction_times']} == set(fit_map), 'all prefix timing rows')
    for row in prefix_rows:
        require(row['attempts'] == saved['dataset_counts']['base']['attempted']
                and row['valid_events'] == saved['dataset_counts']['base']['valid_prefix_events']
                and type(row['mean_nll']) in (int, float) and math.isfinite(row['mean_nll']) and row['mean_nll'] >= 0,
                'finite prefix likelihood with all-attempt denominators')
    require(len(saved['prediction_times']) == 9
            and {(r['arm'], r['seed']) for r in saved['prediction_times']} == set(fit_map), 'all final-model prediction timings')
    for row in saved['prediction_times']:
        require(row['model_state_before'] == row['model_state_after'] == fit_map[row['arm'], row['seed']]['final_state_sha256'],
                'unchanged final model during evaluation')
        require(row['oracle_prefix_input'] is False, 'no oracle boundary input to learned models')
    require(saved['oracle_metadata']['parameter_count'] == 0, 'exact control has zero learned parameters')
    work = saved['structural_work']
    require(work == report['structural_work'] and set(work) == set(ARMS), 'complete independently checked work records')
    for arm in ARMS:
        require(set(work[arm]) == set(ROUTES), 'all training and evaluation work routes')
        for route in ROUTES:
            require(type(work[arm][route]) is dict
                    and bool(work[arm][route])
                    and all(type(value) is int and value >= 0 for value in work[arm][route].values()),
                    'declared nonnegative integer work counters')
    return {'version': VERSION, 'gates': report['gates'], 'rows': rows, 'baseline_rows': baselines,
            'three_seed_means_h4_h8': means, 'data_cases': report['data_cases'],
            'paired_differences_h4_h8': paired,
            'paired_difference_scope': 'Same-seed factorized minus each free control on the same cases. Descriptive signed arithmetic only; no significance, confidence interval or additional criterion.',
            'dataset_counts': saved['dataset_counts'], 'counts': saved['counts'], 'audit_counts': report['counts'],
            'fits': saved['fits'], 'prediction_times': saved['prediction_times'], 'config': plan['config'],
            'prefix_rows': prefix_rows, 'prefix_prediction_times': saved['prefix_prediction_times'],
            'prefix_reconstruction': report['prefix_reconstruction'],
            'dynamics_comparisons': report['dynamics_comparisons'],
            'structural_work': work, 'structural_work_scope': saved['structural_work_scope'],
            'structural_work_validation': report['metadata']['structural_work_validation'],
            'oracle_checks': saved['oracle_checks'], 'oracle_metadata': saved['oracle_metadata'],
            'oracle_state_sha256': saved['oracle_state_sha256'], 'independent_reconstruction': report['target_reconstruction'],
            'generation_seconds': saved['generation_seconds'], 'sources': plan['sources'],
            'phase_seconds': {name: record['terminal']['wall_seconds'] for name, record in phases.items()},
            'total_phase_seconds': math.fsum(record['terminal']['wall_seconds'] for record in phases.values()),
            'phase_timing_scope': 'Original supervisor durations, including launch, execution and process cleanup; qualification includes engineering tests.',
            'audit_limitations': report['limitations'], 'architecture_claim': False,
            'latent_identification_claim': False, 'native_environment_claim': False, 'novelty_claim': False, 'model_selection': False,
            'mean_scope': 'Arithmetic mean of three separately fitted policies, not an ensemble or independent-case confidence interval.'}


def figure(numbers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors, markers = ('#0072B2', '#D55E00', '#009E73'), ('o', 's', '^')
    rows = {(r['arm'], r['seed'], r['horizon']): r for r in numbers['rows']}
    baseline = {r['horizon']: r for r in numbers['baseline_rows']}
    prefix_rows = {(r['arm'], r['seed']): r for r in numbers['prefix_rows']}
    fits = {(r['arm'], r['seed']): r for r in numbers['fits']}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    for ax, metric, title in zip(axes[0], ('blind_regret', 'observed_kl'),
                               ('Blind decision regret', 'Observed filtering KL (nats)'), strict=True):
        for arm, color in zip(ARMS, colors, strict=True):
            values = []
            for seed, marker in zip(SEEDS, markers, strict=True):
                points = [rows[arm, seed, h][metric] for h in HORIZONS]
                values.append(points)
                ax.plot(HORIZONS, points, color=color, marker=marker, markersize=4, linewidth=.8, alpha=.45)
            ax.plot(HORIZONS, [math.fsum(v[i] for v in values) / 3 for i in range(4)], color=color, linewidth=2)
        if metric == 'blind_regret':
            ax.plot(HORIZONS, [baseline[h]['blind_regret'] for h in HORIZONS], color='#333333', linestyle='--')
        ax.axhline(0, color='#666666', linestyle=':', linewidth=.8)
        ax.axvspan(.8, 2.2, color='#DDE4ED', alpha=.28, zorder=0)
        ax.set_title(title)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Forecast horizon (steps)')
        ax.grid(axis='y', alpha=.18)
    for ax, title, lookup, key in ((axes[1, 0], 'All-attempt DEV prefix NLL', prefix_rows, 'mean_nll'),
                                  (axes[1, 1], 'Measured training time per fit', fits, 'seconds')):
        for seed, marker in zip(SEEDS, markers, strict=True):
            values = [lookup[arm, seed][key] for arm in ARMS]
            ax.plot((0, 1, 2), values, color='#777777', linewidth=.7, alpha=.5)
            for index, color in enumerate(colors):
                ax.scatter(index, values[index], color=color, marker=marker, s=35)
        for index, arm in enumerate(ARMS):
            mean = math.fsum(lookup[arm, seed][key] for seed in SEEDS) / 3
            ax.plot((index - .13, index + .13), (mean, mean), color=colors[index], linewidth=3)
        ax.set_xticks((0, 1, 2), ('Factorized', 'Matched free', 'Dense free'))
        ax.set_xlim(-.4, 2.4)
        ax.set_title(title)
        ax.set_ylabel('Nats per valid event' if key == 'mean_nll' else 'Seconds (not compute-matched)')
        ax.grid(axis='y', alpha=.18)
    handles = [Line2D([], [], color=color, linewidth=2, label=label) for color, label in zip(colors, LABELS, strict=True)]
    handles += [Line2D([], [], color='#333333', linestyle='--', label='Uniform reference (regret)'),
                Line2D([], [], color='#666666', linestyle=':', label='Exact-control zero reference')]
    handles += [Line2D([], [], color='#555555', marker=marker, linewidth=.7, label=f'Seed {seed}')
                for seed, marker in zip(SEEDS, markers, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle('Synthetic factorized-dynamics diagnostic: stability across three fit seeds', fontsize=14, fontweight='bold')
    fig.text(.5, .105, 'All three seeds per arm shown. Thick marks: arithmetic means. Shading: trained forecast horizons. '
             'Prefix NLL is descriptive.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.17, wspace=.25, hspace=.34)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    lines = ['# Factorized recurrent dynamics: a synthetic stability diagnostic', '',
        ('All three arms learn the same bounded linear cost head and use the same forecast loss plus '
         'coefficient-one prefix NLL. The intervention changes the dynamics parameterization and its initialization. '
         'Every seed and all three absolute learning criteria remain visible; favorable averages cannot rescue a failed condition.'), '',
        '![All three seeds per arm, decision regret, observed KL, prefix likelihood and fitting cost](benchmark.png)', '',
        ('[Frozen protocol](../finite-factorized-dynamics-protocol.md). Original qualification, fit and audit closures, '
         'source pins and complete payload hashes were authenticated before reading these metrics.'), '',
        '## All three unchanged criteria', '',
        '| Criterion | Factorized | Function-matched free | Dense free |', '|---|---|---|---|']
    for criterion in CRITERIA:
        values = []
        for arm in ARMS:
            result = numbers['gates'][arm][criterion]
            values.append(f"{'PASS' if result['passed'] else 'FAIL'} ({sum(result['conditions'].values())}/{len(result['conditions'])})")
        lines.append('| ' + criterion + ' | ' + ' | '.join(values) + ' |')
    lines += ['', ('Every applicable threshold must hold for all three fit seeds. H1/H2 learning requires blind cost MSE '
        'and regret at most half the positive uniform reference and observed KL at most 0.1 nats. Blind H4/H8 extrapolation '
        'uses the same cost thresholds and H8 survival MAE at most 0.05. Observed H4/H8 filtering requires KL at most 0.1 '
        'and consumes intervening observations. Minimum supports are 256 TRAIN and 64 DEV eligible cases. '
        'The previous replication continuation rule is not part of this new study and remains closed.'), '',
        '## Dynamics and controls', '',
        '| Arm | Dynamics | Trainable parameters | Parameter bytes |', '|---|---|---:|---:|',
        '| factorized | Learned transition T, shared reset/event emission O and action/next-state hazard h | 352 | 2,816 |',
        '| matched_free | Unrestricted B and reset emission, initialized to the factorized function | 1,120 | 8,960 |',
        '| dense_free | Existing unrestricted model and its original dense initialization | 1,120 | 8,960 |', '',
        ('`B[a,o,n,s] = O[o,n] * (1-h[a,n]) * T[a,n,s]`; '
        '`found[a,s] = sum_n h[a,n]*T[a,n,s]`; the blind operator sums ordinary branches. '
        'The factorized reset uses O before any transition or hazard. This imposes the synthetic task\'s conditional '
        'structure; no true numerical transition, emission, hazard or posterior values initialize a learner.'), '',
        ('All arms retain the world-aligned learned-head initialization and known uniform reset prior. Factorization '
         'bundles conditional structure, shared reset/emission, capacity and parameterization. Matched initial functions '
         'do not imply matched gradients. The earlier sticky-initialization study remains a separate failed experiment.'), '',
        '## Audited initial-function checks', '',
        '| Seed | Reset maximum difference | B maximum difference | Found maximum difference | Blind maximum difference | Head maximum difference |',
        '|---|---:|---:|---:|---:|---:|']
    for row in numbers['dynamics_comparisons']['initial_function_checks']:
        errors = row['maximum_absolute_differences']
        lines.append(f"| {row['seed']} | {errors['reset_emission']:.3g} | {errors['observed']:.3g} | "
                     f"{errors['found']:.3g} | {errors['blind']:.3g} | {row['head_maximum_absolute_difference']:.3g} |")
    lines += ['', ('The first four columns compare factorized and matched-free initial probabilities; the head check '
        'covers all three arms. Every difference must be at most 1e-12. Engineering tests separately qualify full '
        'prefix/rollout parity and dense-free parity with the previous model. These are function checks, not a learned '
        'state-recovery or parameter-hash-equivalence claim.'), '',
        '## Longer-horizon means', '',
        '| Arm | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_regret']:.6g} | {row['blind_cost_mse']:.6g} | "
                     f"{row['observed_cost_mse']:.6g} | {row['observed_kl']:.6g} | {row['blind_survival_mae']:.6g} |")
    lines += ['', 'Means average three separately fitted policies on common cases, not an ensemble or a confidence interval.', '',
        '## Every paired contrast against both controls', '',
        ('Differences are factorized minus control on the same seed and cases. Negative favors factorized for that metric. '
         'These are descriptive arithmetic, not significance tests, selection rules or replacements for the absolute criteria.'), '',
        '| Control | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['paired_differences_h4_h8']:
        lines.append(f"| {row['control']} | {row['seed']} | {row['horizon']} | {row['blind_regret']:+.6g} | "
                     f"{row['blind_cost_mse']:+.6g} | {row['observed_cost_mse']:+.6g} | {row['observed_kl']:+.6g} |")
    lines += ['', '## Retained dynamics and readout snapshots', '',
        ('[summary.json](summary.json) retains all 18 initial/final dynamics snapshots in `fits`, including the complete '
         'reset emission, B, found and blind arrays, their canonical little-endian float64 hashes, and each export\'s '
         'separate work record. It also retains all 18 cost matrices and hashes. The table shows head movement only; '
         'movement is not evidence of correct latent-state recovery. Checkpoint/snapshot correspondence remains '
         'source-attested because the numerical audit does not reconstruct models.'), '',
        '| Arm | Seed | Readout mean absolute change | Readout maximum absolute change |', '|---|---:|---:|---:|']
    for row in numbers['dynamics_comparisons']['fits']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['mean_absolute_change']:.6g} | {row['maximum_absolute_change']:.6g} |")
    lines += ['', '## Prefix likelihood and inference cost', '',
        '| Arm | Seed | Attempts | Valid events | Prefix NLL | Prefix seconds | Endpoint seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    prefix_times = {(r['arm'], r['seed']): r['seconds'] for r in numbers['prefix_prediction_times']}
    endpoint_times = {(r['arm'], r['seed']): r['seconds'] for r in numbers['prediction_times']}
    for row in numbers['prefix_rows']:
        key = row['arm'], row['seed']
        lines.append(f"| {row['arm']} | {row['seed']} | {row['attempts']} | {row['valid_events']} | "
                     f"{row['mean_nll']:.6g} | {prefix_times[key]:.6f} | {endpoint_times[key]:.6f} |")
    lines += ['', ('Every attempted public prefix contributes valid events, including its first found event; padding '
        'does not contribute. Only surviving prefixes receive endpoint targets. Learners receive no oracle posterior. '
        'The zero-parameter exact correctness reference alone receives the boundary posterior and true dynamics. '
        'The separate uniform-state reference knows dynamics but discards the prefix.'), '',
        ('Prefix timings include copies, likelihood forwards, guards and scalar NLL, excluding state hashes and file '
         'writes. Endpoint timings include blind, observed and shuffled forecasts with copies and guards, excluding '
         'state hashes and scalar metrics. Neither is single-decision latency.'), '',
        '## Data and complete fitting costs', '',
        '| Split | Attempts | Endpoint eligible | Found-terminated | Valid prefix events |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'base'):
        row = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {row['attempted']} | {row['retained']} | {row['discarded_found']} | {row['valid_prefix_events']} |")
    lines += ['', ('All nine fits use 480 epochs, batch 64, Adam 0.003, clip 5, the same H2 objective and coefficient-one '
        'prefix NLL. Attempt pools and seed-specific batch orders are shared. All final checkpoints precede DEV '
        'generation. There is no warm start, selected seed, horizon extension or loss-weight search.'), '',
        '| Arm | Seed | Parameters | Updates | Endpoint exposures | Prefix-event exposures | Construction seconds | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in numbers['fits']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['parameter_metadata']['parameter_count']} | {row['updates']:,} | "
                     f"{row['training_case_exposures']:,} | {row['training_prefix_event_exposures']:,} | "
                     f"{row['construction_seconds']:.6f} | {row['seconds']:.3f} |")
    lines += ['', ('Construction includes model initialization, matched conversion and guards. Fit time separately '
        'includes initial/final state, head and dynamics snapshots, Adam construction, training, journals, final '
        'checks and checkpoint writing. Arithmetic is float64 except public float32 tokens. All arms have no persistent '
        'buffers; parameter bytes exclude gradients, optimizer state and activations. Equal epochs and fewer '
        'parameters do not establish matched compute or measured speedup.'), '',
        '## Structural work and diagnostic overhead', '',
        '| Arm | Route | Factorizations | Factor products | Free branch softmaxes | Prefix filter rows | Head softmaxes |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for route in ROUTES:
            work = numbers['structural_work'][arm][route]
            free_normalizations = work['operator_observed_softmax_calls'] + work['prefix_operator_softmax_calls']
            lines.append(f"| {arm} | {route} | {work['factorization_calls']:,} | {work['factor_product_entries']:,} | "
                         f"{free_normalizations:,} | {work.get('prefix_filter_rows', 0):,} | {work['cost_head_softmax_calls']:,} |")
    counts = numbers['counts']
    lines += ['', (f"Separate diagnostic totals: {counts['dynamics_snapshot_evaluations']} dynamics exports and "
        f"{counts['readout_snapshot_evaluations']} cost exports, including {counts['readout_snapshot_softmax_evaluations']} "
        'head softmaxes. Dynamics export work is saved per fit and stage, outside the forward-route table. Matched-free '
        'construction separately records one factorization and two log transforms over 1,088 values per fit. '
        'Constructor draws, guards, backward operations, optimizer work and allocations are not exhaustive FLOP counts. '
        'All 11 constructor counters and complete diagnostic/forward work dictionaries remain in the summary.'), '',
        '| Original closed phase | Seconds |', '|---|---:|']
    for phase in ('qualify', 'fit', 'audit'):
        lines.append(f"| {phase} | {numbers['phase_seconds'][phase]:.3f} |")
    lines += [f"| Total of successful phases | {numbers['total_phase_seconds']:.3f} |", '',
        ('Whole-phase times include launch and cleanup and already contain nested costs. Preserve any earlier failed '
         'qualification separately. All 36 endpoint rows, nine prefix rows, full snapshots and work records are retained.'), '',
        '## Interpretation limits', '',
        ('The true dynamics belong to the factorized family, while the exact cost vertices are limiting values of the '
         'shared bounded learned head. Factorization is established modeling, not a novelty claim. Success would concern '
         'the combined structural constraint, capacity and parameterization under this budget; if both initially matched '
         'models succeed, initialization remains a plausible explanation. Four centered costs do not identify all eight '
         'latent coordinates. No convergence, calibration, native transfer, scenario-shift robustness or cross-study '
         'architecture improvement is established by these synthetic measurements. All earlier failed outcomes remain closed.'), '']
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
    receipt = {'version': VERSION, 'registration': {'path': str(plan_path), **descriptor(plan_path)},
        'renderer': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'inputs': {name: {kind: {'path': str(record[kind + '_path']), **descriptor(record[kind + '_path'])}
                          for kind in ('receipt', 'launch', 'terminal')} for name, record in phases.items()},
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
