"""Render only authenticated, closed JSON from the finite cost-readout diagnostic.

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
VERSION = 'finite-cost-readout-report-v1'
ARMS = ('fixed_exact', 'fixed_softened', 'learned_readout')
SEEDS = (424261001, 424261002, 424261003)
HORIZONS = (1, 2, 4, 8)
ROUTES = ('training_blind', 'training_observed', 'training_prefix', 'evaluation_blind', 'evaluation_observed', 'evaluation_shuffled', 'evaluation_prefix')
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Fixed exact C', 'Fixed softened C', 'Learned bounded C')
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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_cost_readout_worker.py'
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
    plan_path = regular(studyfolder / 'registration-01.json')
    plan = read(plan_path)
    require(plan['version'] == 'finite-cost-readout-v1' and plan['root'] == str(ROOT)
            and plan['mode'] == 'study', 'registered cost-readout study')
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
    require(q_plan['mode'] == 'engineering' and q_plan['root'] == str(ROOT)
            and q_plan['sources'] == plan['sources'], 'same qualified source closure')
    qualify = closed_phase(q_plan, q_plan_path, 'qualify')
    require(qualify['receipt_path'] == q_receipt_path and descriptor(qualify['terminal_path']) == q['terminal']
            and qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns'], 'qualification closed before original fit')
    # First result reads happen only after successful original closure and pins.
    saved = read(fit['directory'] / 'summary.json')
    report = read(audit['directory'] / 'audit.json')
    require(saved['version'] == 'finite-cost-readout-v1' and report['version'] == 'finite-cost-readout-audit-v1'
            and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == report['gates'], 'independent audit and explicit external closure boundary')
    require(saved['config'] == plan['config'] and tuple(saved['config']['fit_seeds']) == SEEDS
            and saved['config']['epochs'] == 480 and saved['counts']['fit_count'] == 9
            and saved['checkpoint_barrier']['fit_count'] == 9
            and saved['checkpoint_barrier']['dev_generation_count'] == 0
            and saved['checkpoint_barrier']['oracle_train_verified'] is True, 'complete qualified oracle and fit barrier')
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
    paired = [{'candidate': 'learned_readout', 'control': control, 'seed': seed,
               'regime': 'base', 'horizon': h,
               **{metric: keyed['learned_readout', seed, h][metric] - keyed[control, seed, h][metric]
                  for metric in ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl')}}
              for control in ('fixed_exact', 'fixed_softened') for h in (4, 8) for seed in SEEDS]
    fit_map = {(row['arm'], row['seed']): row for row in saved['fits']}
    for seed in SEEDS:
        for key in ('operator_initial_sha256', 'reset_initial_sha256'):
            require(len({fit_map[arm, seed][key] for arm in ARMS}) == 1, 'paired identical initial filter')
        softened = fit_map['fixed_softened', seed]['readout_initial']['matrix']
        learned = fit_map['learned_readout', seed]['readout_initial']['matrix']
        require(max(abs(softened[i][j] - learned[i][j]) for i in range(4) for j in range(8)) <= 1e-12,
                'softened and learned heads initially function-matched')
    for row in saved['fits']:
        require(row['parameter_metadata']['parameter_count'] == (1120 if row['arm'] == 'learned_readout' else 1088)
                and row['prefix_loss_weight'] == 1, 'declared head capacity and common auxiliary coefficient')
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
            'paired_difference_scope': 'Same-seed learned_readout minus each fixed control on the same cases. Descriptive signed arithmetic only; no significance, confidence interval or additional criterion.',
            'dataset_counts': saved['dataset_counts'], 'counts': saved['counts'], 'audit_counts': report['counts'],
            'fits': saved['fits'], 'prediction_times': saved['prediction_times'], 'config': plan['config'],
            'prefix_rows': prefix_rows, 'prefix_prediction_times': saved['prefix_prediction_times'],
            'prefix_reconstruction': report['prefix_reconstruction'],
            'readout_comparisons': report['readout_comparisons'],
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
        ax.set_xticks((0, 1, 2), ('Fixed exact', 'Fixed softened', 'Learned'))
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
    fig.suptitle('Synthetic cost-readout diagnostic: same filter and loss, different heads', fontsize=14, fontweight='bold')
    fig.text(.5, .105, 'All three fits shown. Thick marks: arithmetic means. Shading: trained forecast horizons. '
             'Prefix NLL is descriptive.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.90, bottom=.17, wspace=.25, hspace=.34)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    lines = ['# Learning the cost readout of a synthetic recurrent filter', '',
        ('Three arms share the same learned reset emission, recurrent branch operators and coefficient-one prefix loss. '
         '`fixed_exact` uses the known centered cost matrix C; `fixed_softened` uses 0.9C; `learned_readout` uses '
         '0.25 minus a softmax over decisions in each latent-state column, initialized to match 0.9C within 1e-12.'), '',
        '![Every fitted seed, endpoint forecasts, prefix likelihood and fitting cost](benchmark.png)', '',
        ('The learned head adds 32 logits, or 24 identifiable head degrees of freedom. Only learned versus softened '
         'isolates trainability from initialization. A gain against softened alone might simply undo the softening; '
         'the exact-C control retains the stronger privileged anchor. All arms start with world-aligned readouts.'), '',
        ('All learned models receive public history only. Every attempted prefix, including its first found event, is '
         'retained for coefficient-one event-mean NLL; post-found padding is excluded. Endpoint targets exist only for '
         'surviving prefixes and use a separate fixed denominator. There is no oracle posterior or belief-supervision '
         'input to a learner. The exact correctness reference alone receives the true boundary posterior and dynamics.'), '',
        ('[Frozen protocol](../finite-cost-readout-protocol.md). Original qualification, fit and audit closures, source '
         'pins and complete payload hashes were authenticated before reading these metrics. Exact-control forecasts '
         'match independently reconstructed targets within 1e-12.'), '',
        '## Unchanged criteria', '',
        '| Criterion | Fixed exact | Fixed softened | Learned readout |', '|---|---|---|---|']
    for criterion in CRITERIA:
        values = []
        for arm in ARMS:
            result = numbers['gates'][arm][criterion]
            values.append(f"{'PASS' if result['passed'] else 'FAIL'} ({sum(result['conditions'].values())}/{len(result['conditions'])})")
        lines.append('| ' + criterion + ' | ' + ' | '.join(values) + ' |')
    lines += ['', ('Every fit seed must pass each criterion and minimum support. H1/H2 learning requires blind cost MSE '
        'and regret at most half the positive uniform reference and observed KL at most 0.1 nats. Blind H4/H8 extrapolation '
        'uses the same cost thresholds and H8 survival MAE at most 0.05. Observed H4/H8 filtering requires KL at most 0.1 '
        'and receives intervening observations. Prefix likelihood cannot rescue these criteria.'), '',
        '## Longer-horizon means', '',
        '| Arm | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_regret']:.6g} | {row['blind_cost_mse']:.6g} | "
                     f"{row['observed_cost_mse']:.6g} | {row['observed_kl']:.6g} | {row['blind_survival_mae']:.6g} |")
    lines += ['', 'Means describe three separately fitted policies, not an ensemble or confidence interval.', '',
        '## Paired differences against both fixed controls', '',
        ('Every difference is learned minus the named control on the same seed and cases. Negative favors learned '
         'for that metric. These are descriptive arithmetic, not significance tests or additional gates.'), '',
        '| Control | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['paired_differences_h4_h8']:
        lines.append(f"| {row['control']} | {row['seed']} | {row['horizon']} | {row['blind_regret']:+.6g} | "
                     f"{row['blind_cost_mse']:+.6g} | {row['observed_cost_mse']:+.6g} | {row['observed_kl']:+.6g} |")
    lines += ['', '## Readout changes for every fit', '',
        '| Arm | Seed | Mean absolute change | Maximum absolute change | Initial SHA256 | Final SHA256 |',
        '|---|---:|---:|---:|---|---|']
    for row in numbers['readout_comparisons']['matrices']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['mean_absolute_change']:.6g} | {row['maximum_absolute_change']:.6g} | "
                     f"{row['initial']['sha256']} | {row['final']['sha256']} |")
    lines += ['', ('The audit checks these saved 4-by-8 matrices and their hashes of little-endian float64 bytes. '
        'Every initial/final matrix is retained in the summary; all learned-head matrices are also printed below. '
        'Matrix movement is not evidence of latent-state recovery or alignment.'), '']
    for row in numbers['readout_comparisons']['matrices']:
        if row['arm'] != 'learned_readout':
            continue
        lines += [f"### Learned head, seed {row['seed']}", '', '| Decision | Initial eight-state costs | Final eight-state costs |',
                  '|---|---|---|']
        for decision in range(4):
            initial = ', '.join(f'{v:.6g}' for v in row['initial']['matrix'][decision])
            final = ', '.join(f'{v:.6g}' for v in row['final']['matrix'][decision])
            lines.append(f'| {decision} | {initial} | {final} |')
        lines.append('')
    lines += ['## Prefix likelihood and inference cost', '',
        '| Arm | Seed | Attempts | Valid events | Prefix NLL | Prefix seconds | Endpoint seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    prefix_times = {(r['arm'], r['seed']): r['seconds'] for r in numbers['prefix_prediction_times']}
    endpoint_times = {(r['arm'], r['seed']): r['seconds'] for r in numbers['prediction_times']}
    for row in numbers['prefix_rows']:
        key = row['arm'], row['seed']
        lines.append(f"| {row['arm']} | {row['seed']} | {row['attempts']} | {row['valid_events']} | "
                     f"{row['mean_nll']:.6g} | {prefix_times[key]:.6f} | {endpoint_times[key]:.6f} |")
    lines += ['', ('Prefix timings include input copies, likelihood forwards, guards, output copies and scalar NLL, excluding '
        'state hashes and file writes. Endpoint timings include blind, observed and shuffled forecasts with copies and guards, '
        'excluding state hashes and scalar metrics. Neither is single-decision latency.'), '',
        '## Data, storage and measured work', '',
        '| Split | Attempts | Endpoint eligible | Found-terminated | Valid prefix events |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'base'):
        row = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {row['attempted']} | {row['retained']} | {row['discarded_found']} | {row['valid_prefix_events']} |")
    lines += ['', ('All nine fits use 480 epochs, batch size 64, Adam 0.003, global gradient clip 5 and the same forecast '
        'objective plus coefficient-one prefix NLL. Paired filter initialization, attempt pools and batch orders are shared. '
        'The learned and softened heads match initial functions and raw filter gradients before clipping; the additional '
        'head gradient can change the global clipping factor. Final checkpoints precede DEV generation.'), '',
        '| Arm | Seed | Parameters | Parameter bytes | Buffer bytes | Updates | Endpoint exposures | Prefix-event exposures | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in numbers['fits']:
        metadata = row['parameter_metadata']
        lines.append(f"| {row['arm']} | {row['seed']} | {metadata['parameter_count']} | {metadata['parameter_bytes']} | "
                     f"{metadata['buffer_bytes']} | {row['updates']:,} | {row['training_case_exposures']:,} | "
                     f"{row['training_prefix_event_exposures']:,} | {row['seconds']:.3f} |")
    lines += ['', ('Fit time includes initial/final state and readout snapshots, optimizer construction, training, journals, '
        'buffer checks and checkpoint writing; model construction is excluded. All arithmetic is float64 except public '
        'float32 tokens. Storage excludes activations and optimizer state. Equal updates are not equal compute.'), '',
        '| Arm | Route | Filter rows | NLL rows | Head softmax calls | Head probability rows | Cost readout rows |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for route in ROUTES:
            work = numbers['structural_work'][arm][route]
            lines.append(f"| {arm} | {route} | {work.get('prefix_filter_rows', 0):,} | {work.get('prefix_nll_rows', 0):,} | "
                         f"{work['cost_head_softmax_calls']:,} | {work['cost_head_probability_rows']:,} | {work.get('cost_readout_rows', 0):,} |")
    counts = numbers['counts']
    lines += ['', (f"Readout snapshots are separately counted: {counts['readout_snapshot_evaluations']} exports, including "
        f"{counts['readout_snapshot_softmax_evaluations']} learned-head softmaxes. They are inside fit timings but outside forward-route counters. "
        'Each learned forward head normalization processes eight latent columns. Work counts are source-constrained '
        'producer attestations checked against schedules and labels; they exclude backward operations, optimizer work, '
        'validation FLOPs and allocation.'), '', '| Original closed phase | Seconds |', '|---|---:|']
    for phase in ('qualify', 'fit', 'audit'):
        lines.append(f"| {phase} | {numbers['phase_seconds'][phase]:.3f} |")
    lines += [f"| Total of successful phases | {numbers['total_phase_seconds']:.3f} |", '',
        ('Whole-phase timings include launch and cleanup; nested timings must not be added again. Any earlier failed '
        'qualification remains separately preserved in the evidence package. [summary.json](summary.json) retains all '
        '36 endpoint rows, nine prefix rows, conditions, per-seed contrasts, readout snapshots and declared work.'), '',
        '## Interpretation limits', '',
        ('The true world is representable with the exact fixed readout, so failure does not prove a fixed-C expressivity '
        'barrier. All arms retain privileged world-aligned initialization. Real-valued learned costs are interior to the '
        'known bounds; float64 can round a cost to a boundary without clipping. Four centered costs do not identify every '
        'coordinate of an eight-state posterior. Lower event NLL or training loss cannot rescue decision criteria, and '
        'observed filtering consumes evidence that blind forecasts do not. No convergence, calibration, architectural '
        'novelty, native transfer or cross-study improvement claim is made. Earlier outcomes remain unchanged.'), '']
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
