"""Render only authenticated, closed JSON from the finite factor diagnostic.

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
VERSION = 'finite-factor-learning-report-v1'
ARMS = ('learned_exact', 'exact_learned', 'learned_learned', 'gru')
SEEDS = (421261001, 421261002, 421261003)
HORIZONS = (1, 2, 4, 8)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Learned prefix / exact operators', 'Exact prefix / learned operators',
          'Learned prefix / learned operators', 'Ordinary GRU')
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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_factor_worker.py'
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
    require(plan['version'] == 'finite-factor-learning-v1' and plan['root'] == str(ROOT)
            and plan['mode'] == 'study', 'registered factor study')
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
    require(saved['version'] == 'finite-factor-learning-v1' and report['version'] == 'finite-factor-learning-audit-v1'
            and report['agreement'] is True and report['exact_oracle_agreement'] is True
            and report['technical_complete'] is False and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and report['latent_identification_claim'] is False
            and audit['receipt']['result'] == report['gates'], 'independent audit and explicit external closure boundary')
    require(saved['config'] == plan['config'] and saved['counts']['fit_count'] == 12
            and saved['checkpoint_barrier']['fit_count'] == 12
            and saved['checkpoint_barrier']['dev_generation_count'] == 0
            and saved['checkpoint_barrier']['oracle_train_verified'] is True, 'complete qualified oracle and fit barrier')
    finite_tree(saved)
    finite_tree(report)
    return plan_path, plan, {'qualify': qualify, 'fit': fit, 'audit': audit}, saved, report


def extract(plan, phases, saved, report):
    rows, baselines = report['rows'], report['baseline_rows']
    expected = {(arm, seed, 'base', h) for arm in ARMS for seed in SEEDS for h in HORIZONS}
    require(len(rows) == 48 and {(r['arm'], r['seed'], r['regime'], r['horizon']) for r in rows} == expected,
            'all four learned arms, three seeds and four horizons')
    require(len(baselines) == 4 and {(r['regime'], r['horizon']) for r in baselines} == {('base', h) for h in HORIZONS},
            'complete uniform-state reference')
    require(len(saved['fits']) == 12 and {(r['arm'], r['seed']) for r in saved['fits']}
            == {(arm, seed) for arm in ARMS for seed in SEEDS}, 'all twelve fit records')
    require(set(report['data_cases']) == {'train', 'base'}, 'exact TRAIN and base DEV support')
    for split, count in report['data_cases'].items():
        require(type(count) is int and count > 0 and count == saved['dataset_counts'][split]['retained'], 'audited retained cases')
    for row in rows:
        require(row['cases'] == report['data_cases']['base']
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in METRICS), 'finite retained model cell')
    require(set(report['gates']) == set(ARMS[:3]), 'three diagnostic factor arms only')
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
    fit_map = {(row['arm'], row['seed']): row for row in saved['fits']}
    require(len(saved['prediction_times']) == 12
            and {(r['arm'], r['seed']) for r in saved['prediction_times']} == set(fit_map), 'all final-model prediction timings')
    for row in saved['prediction_times']:
        require(row['model_state_before'] == row['model_state_after'] == fit_map[row['arm'], row['seed']]['final_state_sha256'],
                'unchanged final model during evaluation')
    require(saved['oracle_metadata']['parameter_count'] == 0, 'exact control has zero learned parameters')
    return {'version': VERSION, 'gates': report['gates'], 'rows': rows, 'baseline_rows': baselines,
            'three_seed_means_h4_h8': means, 'data_cases': report['data_cases'],
            'dataset_counts': saved['dataset_counts'], 'counts': saved['counts'], 'audit_counts': report['counts'],
            'fits': saved['fits'], 'prediction_times': saved['prediction_times'], 'config': plan['config'],
            'oracle_checks': saved['oracle_checks'], 'oracle_metadata': saved['oracle_metadata'],
            'oracle_state_sha256': saved['oracle_state_sha256'], 'independent_reconstruction': report['target_reconstruction'],
            'generation_seconds': saved['generation_seconds'], 'sources': plan['sources'],
            'phase_seconds': {name: record['terminal']['wall_seconds'] for name, record in phases.items()},
            'total_phase_seconds': math.fsum(record['terminal']['wall_seconds'] for record in phases.values()),
            'phase_timing_scope': 'Original supervisor durations, including launch, execution and process cleanup; qualification includes engineering tests.',
            'audit_limitations': report['limitations'], 'architecture_claim': False,
            'latent_identification_claim': False, 'native_environment_claim': False, 'model_selection': False,
            'mean_scope': 'Arithmetic mean of three separately fitted policies, not an ensemble or independent-case confidence interval.'}


def figure(numbers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors, markers = ('#0072B2', '#D55E00', '#009E73', '#CC79A7'), ('o', 's', '^')
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
    fig.suptitle(f"Known / learned factors: {numbers['data_cases']['base']} fresh base DEV cases",
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.5, .16, 'Shaded horizons: trained. Thin lines: every fit. Thick lines: three-fit means. '
             'Exact control checked at all horizons within 1e-12.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, top=.87, bottom=.28, wspace=.24)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    lines = ['# Known and learned factors in a finite observation world', '',
        ('This diagnostic separates learning a useful prefix state from learning transition and observation operators. '
        'The three prespecified criteria remain separate; the ordinary GRU is descriptive.'), '',
        '![Every fitted model and both forecast panels](benchmark.png)', '',
        '| Prefix state | Operators | Arm | Privileged information |', '|---|---|---|---|',
        '| Exact posterior | Exact world | `exact_exact` | Exact prefix posterior and known dynamics; no learned parameters |',
        '| Learned from public history | Exact world | `learned_exact` | Known dynamics and fixed state-basis cost readout |',
        '| Exact posterior | Learned | `exact_learned` | Exact prefix posterior and fixed state-basis cost readout |',
        '| Learned from public history | Learned | `learned_learned` | Fixed state-basis cost readout |', '',
        ('All four factor cells use the same exact linear cost readout. The ordinary GRU is a separate control with learned '
        'prediction heads. Exact posterior inputs are privileged and do not represent a deployable policy.'), '',
        ('The exact/exact control has zero trained parameters. Its five output fields agree with independently reconstructed '
        'targets at every saved TRAIN and DEV horizon within 1e-12. The chart marks the corresponding zero-error reference; '
        'this is a correctness control, not a fitted architecture result.'), '',
        ('[Frozen protocol](../finite-factor-learning-protocol.md). All original process closures, registered source identities '
        'and payload hashes were authenticated before metrics were read.'), '',
        '## Prespecified criteria', '',
        '| Criterion | Learned prefix / exact operators | Exact prefix / learned operators | Both learned |',
        '|---|---|---|---|']
    for criterion in CRITERIA:
        cells = []
        for arm in ARMS[:3]:
            result = numbers['gates'][arm][criterion]
            cells.append(f"{'PASS' if result['passed'] else 'FAIL'} ({sum(result['conditions'].values())}/{len(result['conditions'])})")
        lines.append('| ' + criterion + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', ('Each criterion requires all three fit seeds and minimum support. Short-horizon learning checks H1 and H2 '
        'separately: at most half the uniform-reference cost MSE and regret, plus observed KL at most 0.1. Blind extrapolation '
        'checks the same relative cost criteria separately at H4 and H8, plus H8 survival MAE at most 0.05. Observed filtering '
        'extrapolation checks KL at most 0.1 separately at H4 and H8. It receives intervening observations and is not blind '
        'extrapolation. Relative criteria require a strictly positive reference denominator.'), '',
        ('All condition booleans, all 48 model metric rows, all four uniform-reference rows, and current source pins are in '
        '[summary.json](summary.json). There is no selected seed or replacement arm.'), '',
        '## Three-fit means at longer horizons', '',
        '| Arm | Horizon | Blind regret | Blind cost MSE | Observed KL | Blind survival MAE | Shuffled-prefix regret |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['horizon']} | {row['blind_regret']:.6g} | {row['blind_cost_mse']:.6g} | "
                     f"{row['observed_kl']:.6g} | {row['blind_survival_mae']:.6g} | {row['shuffled_blind_regret']:.6g} |")
    lines += ['', ('Means summarize separately fitted policies; they are not ensemble decisions or confidence intervals. '
        'For the exact-prefix arm, the shuffled view moves the oracle posterior together with its corresponding public prefix. '
        'The uniform-state reference uses known dynamics after discarding the observed prefix; it is not the optimal policy '
        'that ignores history.'), '', '## Data and measured work', '',
        '| Split | Attempted | Retained | Found during prefix |', '|---|---:|---:|---:|']
    for split in ('train', 'base'):
        row = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {row['attempts']} | {row['retained']} | {row['excluded_found']} |")
    lines += ['', ('Excluded prefixes were not replaced. Future found outcomes stay in the targets. Blind predictions retain '
        'unconditional surviving mass; observed predictions condition on the realized history and become absorbing after found.'), '',
        ('All 12 learned models trained for **480 epochs** on H1/H2 targets, with batch size 64, learning rate 0.003 and gradient '
        'clip 5. All final checkpoints preceded DEV generation. This uses ten times the earlier 48-epoch schedule on fresh cases '
        'and different factor controls. It is not a matched causal comparison against that earlier study.'), '',
        '| Arm | Seed | Trainable parameters | Parameter bytes | Updates | Case exposures | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(numbers['fits'], key=lambda item: (ARMS.index(item['arm']), item['seed'])):
        meta = row['parameter_metadata']
        lines.append(f"| {row['arm']} | {row['seed']} | {meta['parameter_count']:,} | {meta['parameter_bytes']:,} | "
                     f"{row['updates']:,} | {row['training_case_exposures']:,} | {row['seconds']:.3f} |")
    lines += ['| exact_exact | deterministic | 0 | 0 | 0 | 0 | Included in generation/exact-control work |', '',
        ('Fit timings include Adam construction, training, durable journals, fixed-buffer checks and final checkpoint writing. '
        'They exclude model construction. Parameter storage excludes buffers, activations and optimizer storage. The factor '
        'arms have different trainable work, so this is not a matched-compute or speedup claim.'), '',
        '| Original closed phase | Seconds |', '|---|---:|']
    for phase in ('qualify', 'fit', 'audit'):
        lines.append(f"| {phase} | {numbers['phase_seconds'][phase]:.3f} |")
    lines.append(f"| Total of these three phases | {numbers['total_phase_seconds']:.3f} |")
    counts = numbers['counts']
    lines += ['', ('These are complete native supervisor durations, including process launch and cleanup. Qualification includes '
        'engineering tests. The scientific fit phase includes exact-control forecasts, generation, training, evaluation and output '
        'writing. The audit reconstructs saved targets and scalar results without model or generator calls.'), '',
        (f"Recorded scientific work: {counts['optimizer_steps']:,} optimizer steps; "
        f"{counts['training_case_exposures']:,} case exposures; {counts['oracle_model_constructions']} untrained exact-control construction; "
        f"{counts['oracle_blind_rollouts']} exact blind and {counts['oracle_observed_rollouts']} exact observed batch rollouts; "
        f"{counts['evaluation_blind_rollouts']} learned blind, {counts['evaluation_observed_rollouts']} observed and "
        f"{counts['evaluation_shuffled_rollouts']} shuffled batch rollouts. All declared work counters are retained in the summary."), '',
        '## Interpretation limits', '',
        ('This is a diagnostic of useful forecast learning under a fixed budget, not a proof of convergence, latent-state recovery '
        'or architectural novelty. The fixed cost readout distinguishes four decision signatures, not every coordinate of an '
        'eight-state posterior. A privileged arm succeeding does not establish native-environment, robotics, Doom, chess or '
        'connectome transfer. Previous study outcomes remain unchanged.'), '']
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
