"""Render authenticated closed JSON only; never import models or generators.

The caller supplies the study directory containing registration-01.json and an
exclusive output directory. Original fit/audit receipts and supervisor closures,
current source pins and opaque payload hashes are checked before report reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('tied_dense', 'untied_dense', 'tied_retentive', 'untied_retentive', 'gru')
SEEDS = (420261001, 420261002, 420261003)
REGIMES, HORIZONS = ('base', 'shift'), (1, 2, 4, 8)
LABELS = ('Tied, dense', 'Untied, dense', 'Tied, retentive', 'Untied, retentive', 'GRU')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')
VERSION = 'finite-observation-learning-report-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.resolve() == path, 'absolute regular original file: ' + str(path))
    return path


def desc(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder,
            'original phase directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no phase symlinks')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = desc(path)
    return result


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'finite saved publication scalar')
    elif type(value) is dict:
        for child in value.values():
            finite_tree(child)
    elif type(value) is list:
        for child in value:
            finite_tree(child)


def closed_phase(plan, plan_path, phase):
    spec = plan['phases'][phase]
    folder, supervision = Path(spec['output']), Path(spec['supervision'])
    require(supervision.name.endswith('.launch.json'), 'original supervisor launch suffix')
    receipt_path = Path(str(folder) + '.receipt.json')
    terminal_path = supervision.with_name(supervision.name[:-len('.launch.json')] + '.terminal.json')
    receipt, launch, terminal = read(receipt_path), read(supervision), read(terminal_path)
    require(receipt['phase'] == phase and receipt['status'] == 'PASS'
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == desc(plan_path)['sha256']
            and receipt['supervision'] == str(supervision) and receipt['output'] == str(folder),
            'original successful registered worker')
    require(receipt['sources_before'] == receipt['sources_after'] == plan['sources'], 'worker source pins unchanged')
    require(receipt['launch'] == launch and all(terminal[key] == value for key, value in launch.items()),
            'original launch/terminal join')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0
            and terminal['group_absent'] is True and terminal['timed_out'] is False
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True,
            'successful original supervisor closure')
    require(terminal['cap_seconds'] == spec['cap_seconds'] and terminal['cwd'] == str(ROOT)
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + spec['cap_seconds'] * 10**9
            and math.isclose(terminal['wall_seconds'],
                             (terminal['finished_ns'] - terminal['started_ns']) / 1e9, rel_tol=0, abs_tol=1e-9),
            'registered native deadline and complete phase timing')
    expected_command = [plan['runtime']['executable'], str(ROOT / 'scripts/finite_observation_worker.py'),
                        '--plan', str(plan_path), '--plan-sha256', desc(plan_path)['sha256'],
                        '--phase', phase, '--supervision', str(supervision), '--output', str(folder)]
    # The worker fixes its exact supervised argv. Flags may have been ordered
    # differently by the owner, so authenticate all unique argument pairs.
    command = terminal['command']
    require(command[:2] == expected_command[:2] and len(command) == len(expected_command)
            and len(set(command[2::2])) == len(command[2::2]),
            'exact original worker and unique registered flags')
    observed_arguments = dict(zip(command[2::2], command[3::2], strict=True))
    expected_arguments = dict(zip(expected_command[2::2], expected_command[3::2], strict=True))
    require(set(observed_arguments) == set(expected_arguments), 'exact registered argument names')
    for flag in ('--plan', '--supervision', '--output'):
        observed_arguments[flag] = str((Path(terminal['cwd']) / observed_arguments[flag]).resolve())
    require(observed_arguments == expected_arguments,
            'exact original worker and registered arguments')
    require(terminal['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and terminal['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'original source-bound supervisor and clock')
    require(inventory(folder) == receipt['files'], 'all original phase payload bytes unchanged')
    return {'directory': folder, 'receipt': receipt, 'terminal': terminal,
            'receipt_path': receipt_path, 'terminal_path': terminal_path, 'launch_path': supervision}


def authenticate(studyfolder):
    plan_path = regular(studyfolder / 'registration-01.json')
    plan = read(plan_path)
    require(plan['root'] == str(ROOT) and plan['mode'] == 'study', 'registered study root and mode')
    for name, expected in plan['sources'].items():
        path = ROOT / name
        require(path.is_relative_to(ROOT) and '..' not in Path(name).parts and desc(path) == expected,
                'current registered source identity: ' + name)
    fit = closed_phase(plan, plan_path, 'fit')
    audit = closed_phase(plan, plan_path, 'audit')
    require(fit['terminal']['finished_ns'] <= audit['terminal']['started_ns']
            and audit['receipt']['producer_receipt'] == desc(fit['receipt_path'])
            and audit['receipt']['producer_terminal'] == desc(fit['terminal_path']), 'original producer closed before audit')
    q = plan['qualification']
    q_receipt_path = regular(Path(q['path']))
    require(desc(q_receipt_path) == q['descriptor'], 'registered qualification receipt')
    q_receipt = read(q_receipt_path)
    q_plan_path = regular(Path(q_receipt['plan']))
    q_plan = read(q_plan_path)
    require(q_plan['mode'] == 'engineering' and q_plan['sources'] == plan['sources'], 'same qualified sources')
    qualify = closed_phase(q_plan, q_plan_path, 'qualify')
    require(qualify['receipt_path'] == q_receipt_path and desc(qualify['terminal_path']) == q['terminal']
            and qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns'], 'original qualification closed before fit')
    # These are the first numerical result JSON reads, after all original closures.
    saved = read(fit['directory'] / 'summary.json')
    report = read(audit['directory'] / 'audit.json')
    require(saved['version'] == 'finite-observation-learning-v1'
            and report['version'] == 'finite-observation-learning-audit-v1'
            and report['agreement'] is True and report['technical_complete'] is False
            and report['requires_original_supervisor_closure'] is True
            and report['architecture_claim'] is False and audit['receipt']['result'] == report['gate'],
            'independent saved-output audit and delegated closure boundary')
    require(saved['config'] == plan['config'] and saved['counts']['fit_count'] == 15
            and saved['checkpoint_barrier']['fit_count'] == 15
            and saved['checkpoint_barrier']['dev_generation_count'] == 0, 'registered complete fit barrier')
    finite_tree(saved)
    finite_tree(report)
    return plan_path, plan, {'qualify': qualify, 'fit': fit, 'audit': audit}, saved, report


def extract(plan, phases, saved, report):
    rows, baselines = report['rows'], report['baseline_rows']
    keys = {(a, s, r, h) for a in ARMS for s in SEEDS for r in REGIMES for h in HORIZONS}
    require(len(rows) == 120 and {(r['arm'], r['seed'], r['regime'], r['horizon']) for r in rows} == keys,
            'complete 120-row model roster')
    require(len(baselines) == 8 and {(r['regime'], r['horizon']) for r in baselines}
            == {(r, h) for r in REGIMES for h in HORIZONS}, 'complete uniform reference roster')
    require(len(saved['fits']) == 15 and {(r['arm'], r['seed']) for r in saved['fits']}
            == {(a, s) for a in ARMS for s in SEEDS}, 'all final fit records')
    require(len(saved['prediction_times']) == 30 and {(r['arm'], r['seed'], r['regime']) for r in saved['prediction_times']}
            == {(a, s, regime) for a in ARMS for s in SEEDS for regime in REGIMES}, 'all prediction timings')
    require(set(report['data_cases']) == {'train', 'base', 'shift'}, 'all dataset supports')
    for split, cases in report['data_cases'].items():
        require(type(cases) is int and cases > 0 and saved['dataset_counts'][split]['retained'] == cases,
                'audited retained support')
    for row in rows:
        require(row['cases'] == report['data_cases'][row['regime']]
                and all(type(row[key]) in (float, int) and math.isfinite(row[key]) for key in METRICS),
                'finite complete model cell')
    for row in baselines:
        require(row['cases'] == report['data_cases'][row['regime']]
                and all(type(row[key]) in (float, int) and math.isfinite(row[key])
                        for key in ('blind_cost_mse', 'blind_regret')), 'finite uniform reference cell')
    for regime in REGIMES:
        gate = report['gate'][regime]
        require(type(gate['passed']) is bool and all(type(v) is bool for v in gate['conditions'].values())
                and gate['passed'] is all(gate['conditions'].values()), 'unchanged full gate conditions')
    fit_keys = {(r['arm'], r['seed']): r for r in saved['fits']}
    for row in saved['prediction_times']:
        require(row['model_state_before'] == row['model_state_after']
                == fit_keys[row['arm'], row['seed']]['final_state_sha256'], 'unchanged final evaluated checkpoint')
    keyed = {(r['arm'], r['seed'], r['regime'], r['horizon']): r for r in rows}
    means = []
    for arm in ARMS:
        for regime in REGIMES:
            for horizon in (4, 8):
                means.append({'arm': arm, 'regime': regime, 'horizon': horizon,
                              'fit_seeds': list(SEEDS), 'cases': report['data_cases'][regime],
                              **{metric: math.fsum(keyed[arm, seed, regime, horizon][metric]
                                                   for seed in SEEDS) / 3 for metric in METRICS}})
    return {'version': VERSION, 'gate': report['gate'], 'rows': rows, 'baseline_rows': baselines,
            'three_seed_means_h4_h8': means, 'data_cases': report['data_cases'],
            'dataset_counts': saved['dataset_counts'], 'counts': saved['counts'],
            'fits': saved['fits'], 'prediction_times': saved['prediction_times'],
            'generation_seconds': saved['generation_seconds'], 'config': plan['config'],
            'phase_seconds': {name: entry['terminal']['wall_seconds'] for name, entry in phases.items()},
            'phase_timing_scope': 'Original native supervisor duration, before process launch through exit and group cleanup.',
            'audit_limitations': report['limitations'], 'architecture_claim': False,
            'robotics_gain_claim': False, 'native_environment_claim': False, 'model_selection': False,
            'mean_scope': 'Arithmetic mean of three separately evaluated fitted policies; no ensemble decisions or independent-seed confidence interval.'}


def figure(numbers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#332288')
    markers = ('o', 's', '^')
    rows = {(r['arm'], r['seed'], r['regime'], r['horizon']): r for r in numbers['rows']}
    baselines = {(r['regime'], r['horizon']): r for r in numbers['baseline_rows']}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True, sharey='row')
    for column, regime in enumerate(REGIMES):
        for row_index, metric in enumerate(('blind_regret', 'observed_kl')):
            ax = axes[row_index, column]
            for arm, color in zip(ARMS, colors, strict=True):
                points = []
                for seed, marker in zip(SEEDS, markers, strict=True):
                    values = [rows[arm, seed, regime, h][metric] for h in HORIZONS]
                    points.append(values)
                    ax.plot(HORIZONS, values, color=color, marker=marker, markersize=3.4,
                            linewidth=.65, alpha=.45)
                mean = [math.fsum(p[i] for p in points) / 3 for i in range(4)]
                ax.plot(HORIZONS, mean, color=color, linewidth=1.8)
            if row_index == 0:
                ax.plot(HORIZONS, [baselines[regime, h]['blind_regret'] for h in HORIZONS],
                        color='#333333', linewidth=1.5, linestyle='--')
                title = 'Base sensing law' if regime == 'base' else 'Shifted sensing law'
                ax.set_title(f"{title} | {numbers['data_cases'][regime]} retained cases", fontsize=12, pad=10)
            else:
                ax.set_xlabel('Forecast horizon (steps)')
            ax.set_xticks(HORIZONS)
            ax.grid(axis='y', alpha=.2)
            ax.margins(x=.04, y=.08)
        axes[0, column].set_ylabel('Blind decision regret (lower is better)')
        axes[1, column].set_ylabel('Observation-law KL (nats; lower is better)')
    handles = [Line2D([], [], color=c, linewidth=2, label=label)
               for label, c in zip(LABELS, colors, strict=True)]
    handles.append(Line2D([], [], color='#333333', linestyle='--', label='Uniform-state reference (regret only)'))
    handles.extend(Line2D([], [], color='#555555', linewidth=.65, marker=marker, markersize=4,
                          label=f'Seed {seed}') for seed, marker in zip(SEEDS, markers, strict=True))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(.5, .014))
    fig.suptitle('Synthetic observation learning diagnostic\nAll five arms and all three fit seeds',
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.5, .117, 'Thin lines and markers: individual fits. Thick lines: arithmetic means. '
             'No native-environment or architecture gain is established.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.08, right=.975, top=.88, bottom=.20, hspace=.27, wspace=.20)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(numbers):
    gate = numbers['gate']
    lines = ['# Synthetic observation learning diagnostic', '',
        (f"**Base: {gate['base']['status']}. Shift: {gate['shift']['status']}.** "
         'These are separate prespecified checks of the tied retentive arm. All five arms remain reported.'), '',
        '![All arms, seeds and horizons](benchmark.png)', '',
        ('This is an eight-state synthetic classification-cost problem. It does not establish an architectural '
         'improvement, robotics gain, native-environment transfer or novelty. Earlier study outcomes remain unchanged.'), '',
        ('All 15 models trained from scratch for 48 epochs on horizons 1 and 2. All final checkpoints were saved '
        'before either DEV set was generated. Evaluation uses horizons 1, 2, 4 and 8. The shifted sensor law is '
         'not announced to the model; its observation KL is descriptive and is not part of the shift gate.'), '',
        ('[Frozen protocol](../finite-observation-learning-protocol.md). '
         'Original source pins, process closures and opaque payload hashes were checked before reading the independent audit.'), '',
        '| Split | Attempted | Retained | Found during prefix |', '|---|---:|---:|---:|']
    for split in ('train', 'base', 'shift'):
        support = numbers['dataset_counts'][split]
        lines.append(f"| {split} | {support['attempts']} | {support['retained']} | {support['excluded_found']} |")
    lines.extend(['', ('Excluded prefixes were not replaced. Future found events remain in the targets. '
        'Blind targets include surviving probability mass without renormalization. Observed targets condition '
        'on each realized history. Observed cost and survival targets become zero after found; '
        'the event-law target is the absorbing found outcome.'), '',
        '| Gate | Passed | Conditions passed | Failed conditions |', '|---|---|---:|---|'])
    for regime in REGIMES:
        record = gate[regime]
        failed = [name for name, passed in record['conditions'].items() if not passed]
        lines.append(f"| {record['status']} | {record['passed']} | "
                     f"{sum(record['conditions'].values())}/{len(record['conditions'])} | "
                     + (', '.join(failed) if failed else 'None') + ' |')
    lines.extend(['', ('The gate checks every tied-retentive fit seed, not only its average. The complete condition '
        'booleans and all 120 model metric rows are retained in [summary.json](summary.json).'), '',
        '## Three-fit means at longer horizons', '',
        '| Arm | Regime | Horizon | Blind regret | Blind cost MSE | Observation KL | Shuffled-prefix regret |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for row in numbers['three_seed_means_h4_h8']:
        lines.append(f"| {row['arm']} | {row['regime']} | {row['horizon']} | {row['blind_regret']:.6g} | "
                     f"{row['blind_cost_mse']:.6g} | {row['observed_kl']:.6g} | {row['shuffled_blind_regret']:.6g} |")
    lines.extend(['', ('Means average the scores of three separate policies. They are not ensemble decisions or '
        'confidence intervals. Regret compares the chosen decision with the minimum exact expected cost. '
        'The uniform-state reference knows the world dynamics but discards the prefix; it is not the optimal '
        'history-ignorant policy because retained prefixes condition the state distribution. Only this reference '
        'uses the frozen 1e-12 tie allowance. Model decisions use raw argmin.'), '',
        ('The prefix shuffle moves the complete public history and its length by a fixed cyclic offset within '
         'each regime. It holds future actions and targets fixed. This is a descriptive sensitivity check.'), '',
        '## Recorded work', '',
        (f"Completed fits: {numbers['counts']['fit_count']}. Optimizer updates: {numbers['counts']['optimizer_steps']}. "
         f"Completed training case exposures: {numbers['counts']['training_case_exposures']}."), '',
        '| Original phase | Wall seconds through cleanup |', '|---|---:|'])
    for phase, seconds in numbers['phase_seconds'].items():
        lines.append(f'| {phase} | {seconds:.3f} |')
    lines.extend(['', '| Target construction | Seconds |', '|---|---:|'])
    for split, seconds in numbers['generation_seconds'].items():
        lines.append(f'| {split} | {seconds:.3f} |')
    lines.extend(['', '| Arm | Seed | Parameters | Parameter bytes | Updates | Fit seconds |',
                  '|---|---:|---:|---:|---:|---:|'])
    for fit in numbers['fits']:
        info = fit['parameter_metadata']
        lines.append(f"| {fit['arm']} | {fit['seed']} | {info['parameter_count']} | {info['parameter_bytes']} | "
                     f"{fit['updates']} | {fit['seconds']:.3f} |")
    lines.extend(['', '| Prediction regime | Sum of 15 prediction-view seconds |', '|---|---:|'])
    for regime in REGIMES:
        seconds = math.fsum(row['seconds'] for row in numbers['prediction_times'] if row['regime'] == regime)
        lines.append(f'| {regime} | {seconds:.3f} |')
    lines.extend(['', ('Whole-phase timings include process startup and cleanup. Fit timers include Adam construction, '
        'training, order/epoch logs and final checkpoint writes; initial model construction and state hashing are '
        'outside those timers. Prediction timers include public tensor copies, all three prediction routes '
        '(blind, observed and shuffled blind), validation and output copies. These components are nested inside '
        'the fit phase and must not be added to it. Shared updates do not imply matched parameter count, '
        'precision or total compute.'), '', '## Limits of the evidence', ''])
    lines.extend('- ' + text for text in numbers['audit_limitations'])
    return '\n'.join(lines) + '\n'


def render(studyfolder, outputfolder):
    studyfolder, outputfolder = Path(studyfolder).resolve(), Path(outputfolder).absolute()
    require(not outputfolder.exists(), 'exclusive new report output')
    plan_path, plan, phases, saved, report = authenticate(studyfolder)
    numbers = extract(plan, phases, saved, report)
    inputs = {str(plan_path): desc(plan_path), str(Path(__file__).resolve()): desc(Path(__file__).resolve())}
    for phase in phases.values():
        for key in ('receipt_path', 'terminal_path', 'launch_path'):
            inputs[str(phase[key])] = desc(phase[key])
    for path in (phases['fit']['directory'] / 'summary.json', phases['audit']['directory'] / 'audit.json'):
        inputs[str(path)] = desc(path)
    outputfolder.mkdir(parents=True, exist_ok=False)
    figure(numbers, outputfolder / 'benchmark.png')
    with (outputfolder / 'summary.json').open('x') as stream:
        json.dump(numbers, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
    with (outputfolder / 'report.md').open('x') as stream:
        stream.write(document(numbers))
    require(all(desc(Path(path)) == value for path, value in inputs.items()), 'publication inputs unchanged during rendering')
    for name, expected in plan['sources'].items():
        require(desc(ROOT / name) == expected, 'source pins unchanged after rendering')
    receipt = {'version': VERSION, 'status': 'completed', 'inputs': inputs,
               'files': {name: desc(outputfolder / name) for name in ('benchmark.png', 'summary.json', 'report.md')},
               'gate': numbers['gate'], 'original_process_closures_authenticated': True,
               'sources_unchanged': True, 'model_selection': False,
               'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0,
               'generator_calls': 0, 'optimizer_calls': 0, 'native_calls': 0}
    with (outputfolder / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--studyfolder', type=Path, required=True)
    parser.add_argument('--outputfolder', type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.studyfolder, args.outputfolder)
    print(json.dumps({'status': receipt['status'], 'output': str(args.outputfolder)}))


if __name__ == '__main__':
    main()
