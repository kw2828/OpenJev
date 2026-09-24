"""Publish only closed, authenticated JSON for time-allocation learning.

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
VERSION = 'finite-training-allocation-report-v1'
STUDY_NAME = 'finite-training-allocation-v1'
REGISTRATION_SHA256 = '2867fc976559808ceda2b21818bdbf394c343f4871a95e6b24bd789a85b10145'
ARMS = ('joint_continuous', 'joint_restart', 'prefix_then_joint')
LABELS = ('Continuous joint', 'Restarted joint', 'Prefix then joint')
SEEDS = (430261001, 430261002, 430261003)
HORIZONS = (1, 2, 4, 8)
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
METRICS = ('blind_cost_mse', 'blind_regret', 'blind_survival_mae', 'observed_cost_mse',
           'observed_survival_mae', 'observed_kl', 'shuffled_blind_regret')
CONFIG = {'seed_namespace': 430260924, 'train_attempts': 512, 'dev_attempts': 128,
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
            and (Path(terminal['cwd']) / command[1]).resolve() == ROOT / 'scripts/finite_training_allocation_worker.py'
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
            and len(plan['sources']) == 47, 'fixed study, configuration and source roster')
    for name, expected in plan['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and descriptor(ROOT / name) == expected, 'unchanged registered source: ' + name)
    # This pinned helper imports only standard-library metadata/clock support.
    import finite_training_allocation_worker as worker

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
            and report['version'] == 'finite-training-allocation-audit-v1'
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
    require(len(advance['conditions']) == 21
            and advance['passed'] is all(advance['conditions'].values()), 'complete allocation advance rule')
    allocation_records, costs = [], []
    for arm in ARMS:
        for seed in SEEDS:
            fit = fits[arm, seed]
            path = auth['phases']['fit']['directory'] / fit['allocation']['path']
            require(descriptor(path) == {k: fit['allocation'][k] for k in ('sha256', 'bytes')},
                    'authenticated allocation JSON')
            allocation = read(path)
            require(allocation['arm'] == arm and allocation['status'] == 'PASS'
                    and allocation['total_seconds'] == 40. and allocation['stage1_seconds'] == 10.
                    and allocation['initial_model_sha256'] == fit['initial_state_sha256']
                    and allocation['final_model_sha256'] == fit['final_state_sha256'], 'exact retained allocation')
            require(fit['parameter_metadata']['parameter_count'] == 352, 'same small model')
            finite_tree(allocation)
            allocation_records.append({'arm': arm, 'seed': seed, 'artifact': fit['allocation'],
                **{k: allocation[k] for k in ('attempted_updates', 'accepted_updates', 'accepted_joint_updates',
                    'accepted_prefix_updates', 'joint_cursor', 'stages', 'boundary', 'checkpoints', 'work',
                    'update_work', 'timed_seconds', 'overrun_seconds', 'final_summary_seconds', 'final_summary_work')}})
            costs.append({'arm': arm, 'seed': seed, 'seconds': fit['seconds'],
                          'timed_seconds': allocation['timed_seconds'],
                          'summary_seconds': allocation['final_summary_seconds'],
                          'overrun_seconds': allocation['overrun_seconds'],
                          'accepted_joint': allocation['accepted_joint_updates'],
                          'accepted_prefix': allocation['accepted_prefix_updates'],
                          'attempted': allocation['attempted_updates']})
    means = [{'arm': arm, 'horizon': h, **{m: math.fsum(keyed[arm, s, h][m] for s in SEEDS) / 3 for m in METRICS}}
             for arm in ARMS for h in HORIZONS]
    log = (auth['phases']['qualify']['directory'] / 'command-1.log').read_text()
    matches = re.findall(r'(?m)^\s*(\d+) passed(?:, [^\n]+)? in [^\n]+$', log)
    require(len(matches) == 1, 'one authenticated test completion')
    return {'version': VERSION, 'config': CONFIG, 'gates': audit['gates'], 'advance': advance,
            'rows': rows, 'three_seed_means': means, 'baseline_rows': audit['baseline_rows'],
            'allocation_comparisons': audit['allocation_comparisons'], 'training_costs': costs,
            'allocations': allocation_records, 'fits': saved['fits'], 'prefix_rows': audit['prefix_rows'],
            'dataset_counts': saved['dataset_counts'], 'scientific_counts': saved['counts'],
            'audit_counts': audit['counts'], 'structural_work': saved['structural_work'],
            'prediction_times': saved['prediction_times'], 'prefix_prediction_times': saved['prefix_prediction_times'],
            'generation_seconds': saved['generation_seconds'], 'engineering_tests_passed': int(matches[0]),
            'phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for arm, color in zip(ARMS, colors, strict=True):
        for seed, marker in zip(SEEDS, markers, strict=True):
            axes[0].plot(HORIZONS, [rows[arm, seed, h]['blind_regret'] for h in HORIZONS],
                         color=color, marker=marker, linewidth=.8, markersize=4, alpha=.4)
        axes[0].plot(HORIZONS, [math.fsum(rows[arm, seed, h]['blind_regret'] for seed in SEEDS) / 3 for h in HORIZONS],
                     color=color, linewidth=2.2)
    axes[0].plot(HORIZONS, [reference[h] for h in HORIZONS], color='#111111', linestyle='--')
    axes[0].axvspan(.8, 2.2, color='#DDE4ED', alpha=.3, zorder=0)
    axes[0].set(title='Decision regret without new observations', xlabel='Forecast horizon', ylabel='Expected cost regret', xticks=HORIZONS)
    for seed, marker in zip(SEEDS, markers, strict=True):
        values = [times[a, seed] for a in ARMS]
        axes[1].plot(range(3), values, color='#888888', linewidth=.7, alpha=.45)
        for i, color in enumerate(colors):
            axes[1].scatter(i, values[i], color=color, marker=marker, s=40)
    for i, arm in enumerate(ARMS):
        mean = math.fsum(times[arm, seed] for seed in SEEDS) / 3
        axes[1].plot((i - .12, i + .12), (mean, mean), color=colors[i], linewidth=3)
        axes[1].annotate(f'{mean:.3f} s', (i, mean), xytext=(0, 12),
                         textcoords='offset points', ha='center', color=colors[i], fontsize=9)
    axes[1].axhline(40., color='#111111', linestyle=':', linewidth=1)
    axes[1].set_xticks(range(3), ('Continuous\njoint', 'Restarted\njoint', 'Prefix then\njoint'))
    axes[1].set(title='Measured training cost, including discarded work', ylabel='Seconds per fit', xlim=(-.4, 2.4))
    for axis in axes:
        axis.grid(axis='y', alpha=.18)
        axis.set_ylim(bottom=0)
    axes[1].set_ylim(0, max(times.values()) * 1.13)
    handles = [Line2D([], [], color=c, linewidth=2, label=label) for c, label in zip(colors, LABELS, strict=True)]
    handles += [Line2D([], [], color='#111111', linestyle='--', label='Known-dynamics uniform reference')]
    handles += [Line2D([], [], color='#777777', marker=m, linewidth=.7, label=f'Seed {s}') for s, m in zip(SEEDS, markers, strict=True)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Same 40-second eligibility window: where should learning time go?', fontsize=14, fontweight='bold')
    fig.text(.5, .14, 'All nine fits shown. Thick marks are means, not an ensemble. Shading: H1/H2 supervision. Costs include overruns.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.25, wspace=.28)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def document(n):
    lines = ['# Recurrent learning under one training-time allowance', '',
        '![Every fit and its measured cost](benchmark.png)', '',
        ('Three methods share the same 352-parameter recurrent model, fresh data and paired initial parameters. '
        'Each has stage deadlines at elapsed 10 and 40 seconds from one original start. '
        'Late updates restore both parameters and optimizer state; all attempted work is retained and charged.'), '',
        '[Protocol](../finite-training-allocation-protocol.md) · [All saved numbers](summary.json) · [Original evidence receipt](receipt.json)', '',
        '| Criterion | Continuous joint | Restarted joint | Prefix then joint |', '|---|---|---|---|']
    for criterion in CRITERIA:
        cells = []
        for arm in ARMS:
            g = n['gates'][arm][criterion]
            cells.append(f"{'PASS' if g['passed'] else 'FAIL'} {sum(g['conditions'].values())}/{len(g['conditions'])}")
        lines.append('| ' + criterion + ' | ' + ' | '.join(cells) + ' |')
    a = n['advance']
    lines += ['', f"**Allocation advance: {'PASS' if a['passed'] else 'FAIL'}, {sum(a['conditions'].values())}/{len(a['conditions'])} conditions.**",
        '', ('Every original criterion must pass. Against each control, all six paired long-horizon regret differences '
        'must be nonpositive, both horizon means must improve at least 10%, and mean measured fit time must be within 5%. '
        'Favorable means do not rescue failed conditions.'), '', '| Advance condition | Outcome |', '|---|---|']
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in a['conditions'].items()]
    lines += ['', '## Every training allocation', '',
        '| Arm | Seed | Accepted joint | Accepted prefix | All attempted | Timed s | Overrun s | Final summary s | Full fit s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in n['training_costs']:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['accepted_joint']} | {r['accepted_prefix']} | {r['attempted']} | "
                     f"{r['timed_seconds']:.6f} | {r['overrun_seconds']:.6f} | {r['summary_seconds']:.6f} | {r['seconds']:.6f} |")
    lines += ['', n['timing_scope'], '', ('The timing columns are nested, not additive. Equal eligibility is not exact '
        'FLOP or wall-time equality. Continuous Adam carries moments; restarted Adam retains the next accepted batch '
        'but discards moments. Prefix training freezes the cost head and starts joint training at batch zero. '
        'Each fit has its own clock, so the joint controls can have different boundary states and batch counts.'), '',
        '## Every endpoint metric', '',
        '| Arm | Seed | H | Cases | Blind MSE | Regret | Survival MAE | Observed MSE | Observed survival MAE | Observed KL | Shuffled regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in n['rows']:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['horizon']} | {r['cases']} | "
                     + ' | '.join(f'{r[k]:.7g}' for k in METRICS) + ' |')
    lines += ['', n['mean_scope'], '', '## Evidence and limits', '',
        (f"The original engineering log reports **{n['engineering_tests_passed']} passing tests**. "
        'Engineering includes the complete independent audit of its saved small run. Science uses fresh histories '
        'and all nine final checkpoints precede DEV generation. The independent audit decodes 27 parameter and '
        '27 optimizer boundaries, reconstructs targets, metrics and work records, and makes no model, optimizer '
        'or environment calls.'), '', '| Original supervised phase | Seconds |', '|---|---:|']
    lines += [f'| {k} | {v:.3f} |' for k, v in n['phase_seconds'].items()]
    lines += ['', ('Complete traces and checkpoints are in the evidence release; this display report keeps their '
        'hashes, boundary records and aggregate work without duplicating every attempted update. Historical '
        'optimizer execution and timings remain source-qualified attestations, not a replay.'), '',
        ('These are synthetic learning results with a privileged initial readout. They do not establish a new '
        'architecture, biological wiring advantage, calibration, scenario-shift robustness or native transfer. '
        'Earlier failed experiments remain closed.'), '']
    return '\n'.join(lines)


def render(study, output, registration_sha256):
    require(registration_sha256 == REGISTRATION_SHA256, 'explicit registered scientific identity')
    output = Path(output)
    require(output == ROOT / 'research/finite-training-allocation-results' and not output.exists(),
            'exclusive exact report directory')
    auth = authenticate(Path(study))
    numbers = extract(auth)
    output.mkdir(exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(numbers, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'report.md').write_text(document(numbers))
    figure(numbers, output / 'benchmark.png')
    require(all(descriptor(Path(p)) == pin for p, pin in auth['inputs'].items()), 'original evidence unchanged')
    for phase in auth['phases'].values():
        require(inventory(phase['directory']) == phase['receipt']['files'], 'complete phase unchanged')
    proof = auth['plan']['kernel_qualification']
    require(inventory(Path(proof['folder'])) == proof['files'], 'numerical prerequisite unchanged')
    receipt = {'version': VERSION, 'registration': {'path': str(auth['plan_path']), **descriptor(auth['plan_path'])},
        'renderer': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
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
