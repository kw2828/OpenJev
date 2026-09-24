"""Render only authenticated, closed JSON from the finite shared-filter diagnostic.

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
VERSION = 'finite-shared-filter-report-v1'
ARMS = ('shared_filter', 'untied_filter', 'gru_prefix')
SEEDS = (422261001, 422261002, 422261003)
HORIZONS = (1, 2, 4, 8)
ROUTES = ('training_blind', 'training_observed', 'evaluation_blind', 'evaluation_observed', 'evaluation_shuffled')
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Shared learned filter', 'Untied learned filter', 'GRU prefix / learned operators')
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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_shared_filter_worker.py'
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
    require(plan['version'] == 'finite-shared-filter-v1' and plan['root'] == str(ROOT)
            and plan['mode'] == 'study', 'registered shared-filter study')
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
    require(saved['version'] == 'finite-shared-filter-v1' and report['version'] == 'finite-shared-filter-audit-v1'
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
    require(set(report['gates']) == set(ARMS[:3]), 'all three diagnostic arms')
    for arm in ARMS[:3]:
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
    paired = [{'candidate': 'shared_filter', 'control': control, 'seed': seed,
               'regime': 'base', 'horizon': h,
               **{metric: keyed['shared_filter', seed, h][metric] - keyed[control, seed, h][metric]
                  for metric in ('blind_regret', 'blind_cost_mse', 'observed_kl')}}
              for control in ('untied_filter', 'gru_prefix') for h in (4, 8) for seed in SEEDS]
    fit_map = {(row['arm'], row['seed']): row for row in saved['fits']}
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
            require(type(work[arm][route]) is dict and work[arm][route]
                    and all(type(value) is int and value >= 0 for value in work[arm][route].values()),
                    'declared nonnegative integer work counters')
    return {'version': VERSION, 'gates': report['gates'], 'rows': rows, 'baseline_rows': baselines,
            'three_seed_means_h4_h8': means, 'data_cases': report['data_cases'],
            'paired_differences_h4_h8': paired,
            'paired_difference_scope': 'Same-seed shared_filter minus control on the same cases. Descriptive signed arithmetic only; no significance, confidence interval or additional criterion.',
            'dataset_counts': saved['dataset_counts'], 'counts': saved['counts'], 'audit_counts': report['counts'],
            'fits': saved['fits'], 'prediction_times': saved['prediction_times'], 'config': plan['config'],
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
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, metric, title in zip(axes, ('blind_regret', 'observed_kl'),
                               ('Blind decision regret', 'Observed filtering KL'), strict=True):
        for arm, color in zip(ARMS, colors, strict=True):
            points = []
            for seed, marker in zip(SEEDS, markers, strict=True):
                values = [rows[arm, seed, h][metric] for h in HORIZONS]
                points.append(values)
                ax.plot(HORIZONS, values, color=color, marker=marker, markersize=3.5, linewidth=.7, alpha=.42)
            ax.plot(HORIZONS, [math.fsum(p[i] for p in points) / 3 for i in range(4)], color=color, linewidth=2)
        if metric == 'blind_regret':
            ax.plot(HORIZONS, [baseline[h]['blind_regret'] for h in HORIZONS], color='#333333', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='#666666', linewidth=1, linestyle=':')
        ax.axvspan(.8, 2.2, color='#DDE4ED', alpha=.28, zorder=0)
        ax.set_title(title + ' (lower is better)', fontsize=12)
        ax.set_xlabel('Forecast horizon (steps)')
        ax.set_ylabel('Expected classification cost gap' if metric == 'blind_regret' else 'KL divergence (nats)')
        ax.set_xticks(HORIZONS)
        ax.grid(axis='y', alpha=.18)
        ax.margins(x=.04, y=.08)
    handles = [Line2D([], [], color=color, linewidth=2, label=label) for color, label in zip(colors, LABELS, strict=True)]
    handles += [Line2D([], [], color='#333333', linestyle='--', label='Uniform state (regret only)'),
                Line2D([], [], color='#666666', linestyle=':', label='Exact-law zero reference')]
    handles += [Line2D([], [], color='#555555', marker=marker, linewidth=.7, markersize=4, label=f'Fit seed {seed}')
                for marker, seed in zip(markers, SEEDS, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.6, bbox_to_anchor=(.5, .005))
    fig.suptitle(f"Synthetic shared-filter diagnostic: {numbers['data_cases']['base']} fresh base DEV cases",
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.5, .16, 'Shaded horizons: trained. Thin lines: every fit. Thick lines: three-fit means. '
             'Exact control checked at all horizons within 1e-12.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, top=.87, bottom=.28, wspace=.24)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    lines = ['# Shared learned filtering in a synthetic observation world', '',
        ('This diagnostic compares learned recurrent prefix filtering with shared versus separate forecast operators, '
         'and the existing GRU-prefix learned-operator model. Every learned arm receives public history only. '
         'The three prespecified criteria remain separate and apply to every arm.'), '',
        '![Every fitted model, seed and forecast horizon](benchmark.png)', '',
        '| Arm | Prefix processing | Forecast operators |', '|---|---|---|',
        '| `shared_filter` | Learned reset emission and recurrent action/odor filtering | Same learned branch operators |',
        '| `untied_filter` | Learned reset emission and separate recurrent action/odor filtering | Independently trainable learned branch operators |',
        '| `gru_prefix` | Existing 31-to-28 GRU and eight-state projection | Learned branch operators |', '',
        ('The two filters condition a uniform reset prior using a learned four-odor emission map. This initial odor '
         'comes before any action or hazard. They then process the eight public action/odor pairs. Only the two filter '
         'arms have matched initial prefix functions; their untied operator parameters start as separate copies. '
         'All three arms have paired initial forecast operators. The GRU prefix is not initially function-matched.'), '',
        ('All learned arms use the same privileged, fixed state-basis cost readout. None receives an oracle prefix '
         'posterior or future hidden state. Only the untrained exact/exact correctness control receives the true prefix '
         'posterior and known dynamics. Its five output fields match reconstructed TRAIN and DEV targets at all saved '
         'horizons within 1e-12. The dotted zero reference represents that control, not a trained result.'), '',
        ('[Frozen protocol](../finite-shared-filter-protocol.md). Original process closures, registered source identities '
         'and payload hashes were authenticated before result metrics were read.'), '',
        '## Prespecified criteria', '',
        '| Criterion | Shared filter | Untied filter | GRU prefix |', '|---|---|---|---|']
    for criterion in CRITERIA:
        cells = []
        for arm in ARMS:
            result = numbers['gates'][arm][criterion]
            cells.append(f"{'PASS' if result['passed'] else 'FAIL'} ({sum(result['conditions'].values())}/{len(result['conditions'])})")
        lines.append('| ' + criterion + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', ('Each criterion requires all three fit seeds and minimum support. At H1/H2, short-horizon learning '
        'requires blind cost MSE and regret at most half the positive uniform reference, and observed KL at most 0.1 nats. '
        'Blind extrapolation applies the cost thresholds at H4/H8 and requires H8 survival MAE at most 0.05. Observed '
        'filtering extrapolation requires KL at most 0.1 separately at H4/H8. It receives intervening observations.'), '',
        ('All condition booleans, all 36 model rows, four uniform-reference rows, paired-fit metadata and current source '
        'pins are retained in [summary.json](summary.json). No seed or arm is selected after evaluation. Absolute criterion '
        'passes alone do not establish superiority of sharing.'), '',
        '## Three-fit means at longer horizons', '',
        '| Arm | Horizon | Blind regret | Blind cost MSE | Observed KL | Blind survival MAE | Shuffled-prefix regret |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_regret']:.6g} | {row['blind_cost_mse']:.6g} | "
                     f"{row['observed_kl']:.6g} | {row['blind_survival_mae']:.6g} | {row['shuffled_blind_regret']:.6g} |")
    lines += ['', '## Paired differences at longer horizons', '',
        ('Each entry is **shared_filter minus the named control**, using the same fit seed and DEV cases. '
         'Negative values favor the shared filter for that metric. These are descriptive arithmetic differences, '
         'not confidence intervals, significance tests, superiority claims or additional passing criteria.'), '',
        '| Control | Seed | Horizon | Blind regret difference | Blind cost MSE difference | Observed KL difference |',
        '|---|---:|---:|---:|---:|---:|']
    for row in numbers['paired_differences_h4_h8']:
        lines.append(f"| {row['control']} | {row['seed']} | {row['horizon']} | {row['blind_regret']:+.6g} | "
                     f"{row['blind_cost_mse']:+.6g} | {row['observed_kl']:+.6g} |")
    lines += ['', ('These means summarize separately fitted policies, not an ensemble or confidence interval. The shuffle '
        'moves complete public prefixes and lengths by one position, with forecast actions and targets unchanged. '
        'The uniform reference uses known dynamics while discarding the prefix; it is not the optimal history-ignorant policy.'), '',
        '## Data and measured work', '',
        '| Split | Attempted | Retained | Found during prefix |', '|---|---:|---:|---:|']
    for split in ('train', 'base'):
        row = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {row['attempts']} | {row['retained']} | {row['excluded_found']} |")
    lines += ['', ('Excluded prefixes were not replaced. Future found cases remain in the targets. Blind predictions carry '
        'unconditional surviving mass. After observed found, cost and survival become zero and the event law is found with '
        'probability one.'), '',
        ('All nine learned fits use 480 epochs, H1/H2 training targets, batch size 64, Adam at 0.003 and gradient clip 5. '
        'The four-part objective combines blind cost MSE, observed cost MSE, half the sum of survival MSEs and observed '
        'event-law soft cross-entropy. There is no posterior-supervision or prefix-likelihood loss. Every final checkpoint '
        'precedes DEV generation. Equal epochs and data do not equalize computation.'), '',
        '| Arm | Seed | Trainable parameters | Parameter bytes | Updates | Case exposures | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(numbers['fits'], key=lambda item: (ARMS.index(item['arm']), item['seed'])):
        meta = row['parameter_metadata']
        lines.append(f"| {row['arm']} | {row['seed']} | {meta['parameter_count']:,} | {meta['parameter_bytes']:,} | "
                     f"{row['updates']:,} | {row['training_case_exposures']:,} | {row['seconds']:.3f} |")
    lines += ['| exact_exact | deterministic | 0 | 0 | 0 | 0 | Included in generation/exact-control work |', '',
        ('Fit timings include optimizer construction, training, journals, buffer checks and checkpoint writes, but exclude '
        'model construction. Parameter bytes exclude buffers, activations and optimizer storage. Filters use float64 '
        'arithmetic; the GRU prefix uses float32 recurrence with float64 projection/operators/readout.'), '',
        '| Arm | Seed | DEV cases | Evaluation seconds |', '|---|---:|---:|---:|']
    for row in numbers['prediction_times']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['cases']} | {row['seconds']:.6f} |")
    lines += ['', ('Evaluation timings include input copies, blind and observed forecasts, the shuffled blind forecast, '
        'guards and prediction copies. They exclude state hashing and scalar metrics, and are not per-decision latency.'), '',
        '## Structural forward work by arm and route', '',
        '| Arm | Route | Filter update rows | GRU update rows | Reset emission rows | Prefix / forecast softmax calls | Transition rows | Conditioned rows | Cost-readout rows |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for route, work in numbers['structural_work'][arm].items():
            lines.append(f"| {arm} | {route} | {work.get('prefix_filter_rows', 0):,} | "
                         f"{work.get('prefix_assimilation_rows', 0):,} | {work.get('reset_emission_rows', 0):,} | "
                         f"{work.get('prefix_operator_softmax_calls', 0):,} / {work.get('operator_observed_softmax_calls', 0):,} | "
                         f"{work.get('blind_transition_rows', 0):,} | {work.get('observed_conditioning_rows', 0):,} | "
                         f"{work.get('cost_readout_rows', 0):,} |")
    lines += ['', ('Counts aggregate all three fits and expose the separate prefix and forecast operator normalizations. '
        'They are source-constrained producer attestations, with fixed geometry and observed row totals checked by the audit. '
        'They exclude validation FLOPs, backward operations, optimizer work and allocations. Full counters are retained in '
        'the summary; these are not FLOP or matched-compute measurements.'), '',
        '| Original closed phase | Seconds |', '|---|---:|']
    for phase in ('qualify', 'fit', 'audit'):
        lines.append(f"| {phase} | {numbers['phase_seconds'][phase]:.3f} |")
    lines.append(f"| Total of these three phases | {numbers['total_phase_seconds']:.3f} |")
    counts = numbers['counts']
    lines += ['', ('These are complete original supervisor durations, including launch and cleanup. Qualification includes '
        'engineering tests. Fit includes data generation, exact controls, learning, evaluation and output writes. Nested fit '
        'and inference timings must not be added again to these totals.'), '',
        (f"Recorded work: {counts['optimizer_steps']:,} optimizer steps; {counts['training_case_exposures']:,} case exposures; "
         f"{counts['oracle_model_constructions']} untrained exact-control construction; {counts['oracle_blind_rollouts']} exact blind "
         f"and {counts['oracle_observed_rollouts']} exact observed rollouts; {counts['evaluation_blind_rollouts']} learned blind, "
         f"{counts['evaluation_observed_rollouts']} observed and {counts['evaluation_shuffled_rollouts']} shuffled batch rollouts."), '',
        '## Interpretation limits', '',
        ('This is a finite synthetic diagnostic of learned filtering under a fixed budget. The shared versus untied comparison '
        'tests parameter sharing within this model class; the GRU comparison also changes encoder structure, precision and '
        'capacity. Classical filtering structure is not a novelty claim. Known C and the filter reset prior remain privileged '
        'task assumptions. Four centered decision costs do not identify every coordinate of an eight-state posterior. '
        'There is no convergence, latent-recovery, architectural-superiority, native-environment, robotics, Doom, chess or '
        'connectome claim. Earlier study outcomes are unchanged; this is not a matched cross-study improvement claim.'), '']
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
