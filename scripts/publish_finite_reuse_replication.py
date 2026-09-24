"""Publish closed, audited saved JSON and opaque evidence without replaying learning."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/finite-reuse-replication-v1'
PLAN = STUDY / 'study-registration.json'
PLAN_SHA = '2b456de9caf180c6f7505a584d91fa9c58fb4222b8848b24f73363994e426f21'
ENGINEERING_SHA = 'caff144669cb0e52d4fabbc43d4734929f74b04e21d1e00fc2ebed953d2ea27a'
OUTPUT = ROOT / 'research/finite-reuse-replication-results'
OVERVIEW = ROOT / 'research/finite-reuse-replication-results.md'
VERSION = 'finite-reuse-replication-publication-v1'
ARMS = ('original_free', 'matched_free', 'rounded')
SEEDS = (435261001, 435261002, 435261003, 435261004, 435261005)
REGIMES = ('base', 'shift')
CRITERIA = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION',
            'OBSERVED_FILTERING_EXTRAPOLATION')
LABELS = ('Original free', 'Matched free', 'Rounded')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
            'canonical ordinary evidence file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def authenticate():
    """Authenticate all original processes and bytes before current metric JSON."""
    require(descriptor(PLAN)['sha256'] == PLAN_SHA, 'exact scientific registration')
    plan = read(PLAN)
    require(len(plan['sources']) == 116 and all(descriptor(ROOT / name) == expected
            for name, expected in plan['sources'].items()), 'all frozen sources before helper imports')
    from finite_reuse_replication_worker import (
        admit_qualification,
        closed_producer,
        files,
        validate_launch_binding,
        validate_registered_plan,
    )
    validate_registered_plan(PLAN, plan)
    engineering_path = STUDY / 'engineering-registration-01.json'
    require(descriptor(engineering_path)['sha256'] == ENGINEERING_SHA,
            'exact original engineering registration')
    require(sorted(STUDY.glob('engineering-registration-*.json')) == [engineering_path],
            'one original engineering attempt')
    qualify_terminal = admit_qualification(plan, plan['qualification']['path'])
    require(descriptor(plan['qualification']['path']) == plan['qualification']['descriptor']
            and descriptor(qualify_terminal) == plan['qualification']['terminal'],
            'original qualification joins scientific plan')
    phases = {}
    for phase in ('qualify', 'fit', 'audit'):
        spec = plan['phases'][phase]
        receipt_path = Path(spec['output'] + '.receipt.json')
        receipt = read(receipt_path)
        original_plan = engineering_path if phase == 'qualify' else PLAN
        require(receipt['phase'] == phase and receipt['status'] == 'PASS'
                and receipt['plan'] == str(original_plan)
                and receipt['plan_sha256'] == descriptor(original_plan)['sha256']
                and receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
                'original successful phase and unchanged complete sources')
        validate_launch_binding(plan, receipt)
        require(read(spec['supervision']) == receipt['launch'], 'persisted original launch')
        terminal_path = closed_producer(receipt)
        require(files(spec['output']) == receipt['files'], 'complete original phase payload inventory')
        phases[phase] = {'receipt': receipt, 'receipt_path': receipt_path,
                         'terminal': read(terminal_path), 'terminal_path': terminal_path}
    for before, after in (('qualify', 'fit'), ('fit', 'audit')):
        first, second = phases[before]['terminal'], phases[after]['terminal']
        require(first['clock_backend'] == second['clock_backend']
                and first['finished_ns'] <= second['started_ns'], 'original phase chronology')
    require(phases['audit']['receipt']['producer_receipt'] == descriptor(phases['fit']['receipt_path'])
            and phases['audit']['receipt']['producer_terminal'] == descriptor(phases['fit']['terminal_path']),
            'audit binds the original closed producer')
    require(phases['fit']['receipt']['result'] == {'fits': 15, 'rows': 120}, 'all fifteen fits and endpoint rows')
    study_files = files(STUDY)
    # All current scientific results are first read after the checks above.
    producer = read(Path(plan['phases']['fit']['output']) / 'summary.json')
    audit = read(Path(plan['phases']['audit']['output']) / 'audit.json')
    from audit_finite_observation_learning import compare
    require(producer['version'] == 'finite-reuse-replication-v1' and producer['implementation'] == 'reuse'
            and producer['config'] == plan['config'] and audit['version'] == 'finite-reuse-replication-audit-v1'
            and audit['profile'] == 'science' and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['exact_oracle_agreement'] is True, 'original independently audited scientific record')
    expected_fits = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(producer['fits']) == 15 and {(r['arm'], r['seed']) for r in producer['fits']} == expected_fits,
            'complete fixed fit roster')
    require(len(audit['rows']) == 120 and {(r['regime'], r['arm'], r['seed'], r['horizon']) for r in audit['rows']}
            == {(regime, arm, seed, h) for regime in REGIMES
                for arm, seed in expected_fits for h in (1, 2, 4, 8)}, 'complete separate-regime metric roster')
    row_key = lambda row: (row['regime'], row['arm'], row['seed'], row['horizon'])
    compare(sorted(producer['rows'], key=row_key), sorted(audit['rows'], key=row_key))
    baseline_key = lambda row: (row['regime'], row['horizon'])
    compare(sorted(producer['baseline_rows'], key=baseline_key), sorted(audit['baseline_rows'], key=baseline_key))
    compare(producer['prefix_rows'], audit['prefix_rows'])
    require(len(audit['prefix_rows']) == 30 and producer['structural_work'] == audit['structural_work'],
            'all prefix rows and disjoint actual work')
    require(audit['metadata']['fits'] == 15 and audit['metadata']['payloads'] == 157
            and audit['counts'] == {'array_decodes': 89, 'checkpoint_decodes': 45,
                'optimizer_json_decodes': 45, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0}, 'complete saved-output audit scope')
    require(set(audit['gates']) == set(audit['allocation_comparisons']) == set(REGIMES),
            'both development regimes without pooling')
    combined_conditions = {}
    for regime in REGIMES:
        require(set(audit['gates'][regime]) == set(ARMS), 'all three arms in each regime')
        for arm in ARMS:
            require(set(audit['gates'][regime][arm]) == set(CRITERIA), 'all three absolute criteria')
            for gate in audit['gates'][regime][arm].values():
                require(all(type(v) is bool for v in gate['conditions'].values())
                        and gate['passed'] == all(gate['conditions'].values()), 'saved absolute gate conjunction')
        comparison = audit['allocation_comparisons'][regime]
        advance = comparison['advance']
        require(comparison['regime'] == regime and len(comparison['paired']) == 20
                and len(comparison['mean_regret']) == 4 and len(comparison['mean_fit_time']) == 2
                and len(advance['conditions']) == 27
                and all(type(v) is bool for v in advance['conditions'].values())
                and advance['passed'] == all(advance['conditions'].values())
                and advance['status'] == 'UPDATE_MATCHED_ADVANCE_' + ('PASS' if advance['passed'] else 'FAIL'),
                'all twenty-seven binding regime conditions')
        combined_conditions.update({regime + '/' + key: value for key, value in advance['conditions'].items()})
    require(audit['allocation_comparisons']['base']['mean_fit_time']
            == audit['allocation_comparisons']['shift']['mean_fit_time'],
            'the same fifteen fits supply both evaluation regimes')
    advance = audit['advance']
    require(len(combined_conditions) == 54 and advance['conditions'] == combined_conditions
            and advance['passed'] == all(combined_conditions.values())
            and advance['status'] == 'REPLICATION_SHIFT_ADVANCE_' + ('PASS' if advance['passed'] else 'FAIL')
            and phases['audit']['receipt']['result'] == {'gates': audit['gates'], 'advance': advance},
            'both regimes must pass without pooled rescue')
    require(producer['counts']['fit_count'] == 15 and producer['counts']['accepted_prefix_steps'] == 15360
            and producer['counts']['accepted_joint_steps'] == 46080
            and producer['counts']['accepted_optimizer_steps'] == 61440
            and producer['checkpoint_barrier']['dev_generation_count'] == 0
            and producer['counts']['dev_generation_count'] == producer['counts']['oracle_model_constructions'] == 2,
            'exact scientific update counts and pre-DEV barrier')
    log = (Path(plan['phases']['qualify']['output']) / 'command-1.log').read_text()
    completions = re.findall(r'(?m)^\s*(\d+) passed(?:, (\d+) warnings?)? in [^\n]+$', log)
    require(completions == [('336', '1')], 'exact original 336-test passing qualification and one warning')
    qualification = {'tests_passed': int(completions[0][0]), 'warnings': int(completions[0][1] or 0)}
    return {'plan': plan, 'phases': phases, 'producer': producer, 'audit': audit,
            'study_files': study_files, 'qualification': qualification}



def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.6, 7.4), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, .95))
    colors = ('#2864a3', '#c17622', '#248372', '#9264a8', '#bd4957')
    labels = ('Original\nfree', 'Matched\nfree', 'Rounded')
    keyed = {(r['regime'], r['arm'], r['seed'], r['horizon']): r for r in summary['rows']}
    fitted = {(r['arm'], r['seed']): r for r in summary['fits']}
    axes = []
    for row, regime in enumerate(REGIMES):
        for column, horizon in enumerate((4, 8)):
            axis = fig.add_subplot(grid[row, column])
            axes.append(axis)
            values = {(arm, seed): keyed[regime, arm, seed, horizon]['blind_regret']
                      for arm in ARMS for seed in SEEDS}
            for j, seed in enumerate(SEEDS):
                axis.plot([i + (j - 2) * .04 for i in range(3)], [values[arm, seed] for arm in ARMS],
                          color=colors[j], marker='o', linewidth=1, markersize=4.5,
                          alpha=.9, label=str(seed))
            for i, arm in enumerate(ARMS):
                axis.scatter(i, math.fsum(values[arm, seed] for seed in SEEDS) / len(SEEDS),
                             color='#152130', marker='D', s=32, zorder=4,
                             label='Arithmetic mean' if i == 0 else None)
            if all(value > 0 for value in values.values()):
                axis.set_yscale('log')
                axis.set_ylabel('Regret, log scale (lower is better)', fontsize=9)
            else:
                axis.set_ylim(bottom=0)
                axis.set_ylabel('Regret (lower is better)', fontsize=9)
            epsilon = summary['regimes'][regime]['epsilon']
            axis.set_title(f'{regime.upper()}, ε={epsilon:.2f}: H{horizon}', fontsize=11)
    cost_axis = fig.add_subplot(grid[:, 2])
    axes.append(cost_axis)
    for j, seed in enumerate(SEEDS):
        cost_axis.plot([i + (j - 2) * .04 for i in range(3)],
                       [fitted[arm, seed]['seconds'] for arm in ARMS],
                       color=colors[j], marker='o', linewidth=1, markersize=4.5, alpha=.9)
    for i, arm in enumerate(ARMS):
        cost_axis.scatter(i, math.fsum(fitted[arm, seed]['seconds'] for seed in SEEDS) / len(SEEDS),
                          color='#152130', marker='D', s=32, zorder=4)
    cost_axis.set_ylim(bottom=0)
    cost_axis.set_title('Complete fit time\nEach model trained once', fontsize=11)
    cost_axis.set_ylabel('Seconds (lower is faster)', fontsize=9)
    for axis in axes:
        axis.set_xticks(range(3), labels, fontsize=9)
        axis.set_xlim(-.3, 2.3)
        axis.tick_params(axis='y', labelsize=9)
        axis.grid(axis='y', color='#e4e8ed')
        axis.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Five fresh seeds: base replication and observation-noise shift', fontsize=14, weight='bold')
    fig.savefig(path, dpi=180, metadata={'Description':
        'Saved audited JSON only. Every model and fit seed in both regimes; paired lines identify seeds. '
        'Arithmetic means, no confidence intervals. Positive regret panels use explicitly labeled log axes; '
        'a panel containing zero uses a linear axis. Full fit time is shown once, not once per evaluation regime.'})
    plt.close(fig)


def document(summary):
    audit = summary['audit']
    lines = ['# Fresh replication and observation-noise shift', '',
        f"**{summary['status']}**. {sum(audit['advance']['conditions'].values())}/54 prospectively fixed continuation conditions pass.", '',
        ('Fifteen fresh fits use five seeds, three models and the same qualified per-call reuse implementation. '
         'Each fit completes 1,024 prefix updates followed by 3,072 joint updates, with same-seed minibatches paired '
         'across arms. Every final checkpoint precedes both evaluation cohorts. Each model is evaluated unchanged '
         'on BASE and SHIFT; it is not trained twice.'), '',
        '![All five seeds in both regimes, with complete fit times](benchmark.png)', '',
        ('BASE uses observation error ε=0.12; SHIFT uses ε=0.30. The correct-odor probability changes from '
         '0.88 to 0.70 and each other odor from 0.04 to 0.10. Transition, hazard, costs and action distributions '
         'stay fixed. Independent case streams make this an observation-noise transfer test, not paired noise '
         'perturbations of the same histories or a challenge to the favorable transition prior.'), '',
        '| Regime | Continuation | Conditions passed | Retained DEV / attempted |',
        '|---|---|---:|---:|']
    for regime in REGIMES:
        gate = audit['allocation_comparisons'][regime]['advance']
        lines.append(f"| {regime.upper()} | {'PASS' if gate['passed'] else 'FAIL'} | {sum(gate['conditions'].values())}/27 | {summary['data_cases'][regime]}/512 |")
    lines += ['', ('Neither regime rescues the other. Every paired fit and absolute condition remains binding; '
                  'there is no pooling or mean-based rescue.'), '',
        '| Regime | Model | Short horizon | Blind extrapolation | Observed filtering |',
        '|---|---|---:|---:|---:|']
    for regime in REGIMES:
        for arm, label in zip(ARMS, LABELS, strict=True):
            values = []
            for criterion in CRITERIA:
                gate = audit['gates'][regime][arm][criterion]
                values.append(f"{'PASS' if gate['passed'] else 'FAIL'} {sum(gate['conditions'].values())}/{len(gate['conditions'])}")
            lines.append('| ' + ' | '.join([regime.upper(), label, *values]) + ' |')
    lines += ['', '| Regime | Control | H | Rounded mean regret | Control mean regret | Relative reduction |',
        '|---|---|---:|---:|---:|---:|']
    for regime in REGIMES:
        for row in audit['allocation_comparisons'][regime]['mean_regret']:
            reduction = 'undefined (zero control)' if row['relative_reduction'] is None else f"{100 * row['relative_reduction']:.2f}%"
            lines.append(f"| {regime.upper()} | {row['control']} | {row['horizon']} | {row['candidate_mean_regret']:.8g} | {row['control_mean_regret']:.8g} | {reduction} |")
    lines += ['', ('Relative reductions are ratios of the five fit means; positive values favor rounded. '
        'Seed-level values and paired differences below remain essential: a large mean reduction can be driven '
        'by a few weak control fits. Five seeds share one training cohort and do not constitute five independent '
        'population studies. No confidence interval or significance claim is made. The figure labels log axes '
        'where every regret is positive; any panel with zero uses a linear axis. Diamonds show arithmetic means.'), '',
        '| Control | Rounded mean fit seconds | Control mean fit seconds | Rounded / control time |',
        '|---|---:|---:|---:|']
    for row in audit['allocation_comparisons']['base']['mean_fit_time']:
        lines.append(f"| {row['control']} | {row['candidate_mean_seconds']:.6f} | {row['control_mean_seconds']:.6f} | {row['ratio']:.6f} |")
    lines += ['', '| Model | Fit seed | Prefix updates | Joint updates | Prefix stage seconds | Joint stage seconds | Complete fit seconds | Controller seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    stages = {(r['arm'], r['seed']): r['stages'] for r in audit['allocation']['stages']}
    for row in sorted(summary['fits'], key=lambda r: (ARMS.index(r['arm']), r['seed'])):
        stage_seconds = [s['stopped_elapsed'] - s['start_elapsed'] for s in stages[row['arm'], row['seed']]]
        lines.append(f"| {row['arm']} | {row['seed']} | {row['accepted_prefix_updates']} | {row['updates']} | {stage_seconds[0]:.6f} | {stage_seconds[1]:.6f} | {row['seconds']:.6f} | {row['timed_seconds']:.6f} |")
    times, qualification = summary['phase_seconds'], summary['qualification']
    lines += ['', (f"Original native phases: qualification {times['qualify']:.6f}s, fit/evaluation {times['fit']:.6f}s, "
        f"audit {times['audit']:.6f}s. Selected qualification: {qualification['tests_passed']} passing tests, "
        f"{qualification['warnings']} warning(s), plus the retained independently audited exposure probe."), '',
        ('Complete fit time includes construction, validation, updates, boundary checkpoints and allocation-log '
         'persistence. Stage times and controller time are nested; the controller includes its final summary. '
         'These components must not be added again to the enclosing native phase. The same fit costs apply to '
         'both evaluation regimes and are shown once. Timing is descriptive, not a continuation condition. '
         'Equal update counts do not imply equal time, FLOPs, gradients or effective degrees of freedom.'), '',
        (f"TRAIN retains {summary['data_cases']['train']} of 512 attempted prefixes. All attempted histories, "
         'including terminal found events, contribute to prefix likelihood; endpoint metrics use surviving '
         'prefixes without replacement. Training endpoint labels cover H1/H2; H4/H8 are extrapolation. '
         'Observed filtering receives intervening observations, while blind extrapolation does not.'), '',
        ('The known doubly stochastic world structure and privileged starting readouts favor the tested family. '
         'The observation-noise shift preserves both. Learned models receive public histories and supervised '
         'targets, not exact oracle boundary states, noise-level hints or SHIFT training. Separate ε-bound '
         'zero-parameter references validate the correct world in each regime.'), '',
        ('The independent audit reconstructs saved targets, metrics, schedules, all 45 model and 45 Adam '
         'boundaries, and disjoint forward work. It does not replay learning. The publication reads saved JSON '
         'and hashes opaque evidence; its zero learner calls do not describe the training phase. Shared forward '
         'operations are counted once, separately from logical rollouts and unmeasured backward FLOPs.'), '',
        ('The prior local 19-condition pass, prior equal-time failure and feasibility stops remain closed. '
         'This new result must stand on its own 54 conditions. It does not establish ICLR readiness, calibrated '
         'text probabilities, biological benefit, latent identification, native transfer or architectural novelty. '
         'Before a broader claim, a separate frozen study must challenge the favorable transition prior and '
         'privileged initialization, followed by a second environment. No follow-up is executed or admitted here.'), '',
        'Every continuation condition:', '', '| Condition | Result |', '|---|---|']
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in audit['advance']['conditions'].items())
    lines += ['', 'Every failed absolute condition:', '']
    failures = [f"- {regime.upper()} / {arm} / {criterion}: {name}"
        for regime in REGIMES for arm in ARMS for criterion in CRITERIA
        for name, value in audit['gates'][regime][arm][criterion]['conditions'].items() if not value]
    lines.extend(failures or ['None.'])
    lines += ['', 'Every paired H4/H8 regret comparison:', '',
        '| Regime | Control | Seed | H | Rounded regret | Control regret | Rounded minus control |',
        '|---|---|---:|---:|---:|---:|---:|']
    for regime in REGIMES:
        for row in audit['allocation_comparisons'][regime]['paired']:
            value = row['blind_regret']
            lines.append(f"| {regime.upper()} | {row['control']} | {row['seed']} | {row['horizon']} | {value['candidate']:.8g} | {value['control']:.8g} | {value['difference']:.8g} |")
    lines += ['', 'All endpoint results:', '',
        '| Regime | Model | Seed | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE | Observed survival MAE | Shuffled regret |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl',
               'blind_survival_mae', 'observed_survival_mae', 'shuffled_blind_regret')
    for row in sorted(summary['rows'], key=lambda r: (REGIMES.index(r['regime']), ARMS.index(r['arm']), r['seed'], r['horizon'])):
        lines.append(f"| {row['regime'].upper()} | {row['arm']} | {row['seed']} | {row['horizon']} | " +
                     ' | '.join(f'{row[key]:.8g}' for key in metrics) + ' |')
    lines += ['', 'All-attempt prefix diagnostics (descriptive):', '',
        '| Regime | Model | Seed | Attempts | Valid events | Mean event NLL |',
        '|---|---|---:|---:|---:|---:|']
    for row in summary['prefix_rows']:
        lines.append(f"| {row['regime'].upper()} | {row['arm']} | {row['seed']} | {row['attempts']} | {row['valid_events']} | {row['mean_nll']:.8g} |")
    lines += ['', ('[Protocol](../finite-reuse-replication-protocol.md) · [Complete saved summary](summary.json) · '
        '[Current-study evidence](evidence.tar.gz) · [Manifest](manifest.json) · [Publication receipt](receipt.json)'), '',
        ('The archive contains this complete current study, both 116-source snapshots, its engineering exposure '
         'probe, original closures, all predictions and model/optimizer boundaries, and these publication artifacts. '
         'The parent local pass and earlier linked studies, interpreter and installed packages remain external. '
         'Their registered evidence is reauthenticated and its descriptors remain in the plans and summary.'), '']
    return '\n'.join(lines)

def archive(output, study_files):
    from finite_reuse_replication_worker import publish
    paths = {'study/' + name: (STUDY / name, pin) for name, pin in study_files.items()}
    for name in ('summary.json', 'report.md', 'benchmark.png'):
        paths['publication/' + name] = (output / name, descriptor(output / name))
    paths['publication/overview.md'] = (OVERVIEW, descriptor(OVERVIEW))
    paths['publication/publisher.py'] = (Path(__file__).resolve(), descriptor(Path(__file__).resolve()))
    manifest = {'version': VERSION, 'registration': {'path': str(PLAN), **descriptor(PLAN)},
                'files': {name: {'original_path': str(path), **pin} for name, (path, pin) in sorted(paths.items())},
                'scope': 'Complete current study and this publication; earlier linked evidence and installed runtime remain external.',
                'self_exclusion': 'Manifest is included in the archive but excludes its own hash.'}
    publish(output / 'manifest.json', manifest)
    paths['publication/manifest.json'] = (output / 'manifest.json', descriptor(output / 'manifest.json'))
    with ((output / 'evidence.tar.gz').open('xb') as raw,
          gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped,
          tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as saved):
        for name, (path, pin) in sorted(paths.items()):
            require(descriptor(path) == pin, 'unchanged archive input')
            data = path.read_bytes()
            require({'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} == pin,
                    'opaque archive bytes match pin')
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            saved.addfile(info, io.BytesIO(data))
    with tarfile.open(output / 'evidence.tar.gz', 'r:gz') as saved:
        require(saved.getnames() == sorted(paths), 'complete exact archive roster')
        for member in saved.getmembers():
            path, pin = paths[member.name]
            require(member.isfile() and saved.extractfile(member).read() == path.read_bytes()
                    and descriptor(path) == pin, 'opaque archive byte roundtrip')
    return len(paths)


def main():
    require(not OUTPUT.exists() and not OVERVIEW.exists(), 'exclusive publication destinations')
    publisher_pin = descriptor(Path(__file__).resolve())
    auth = authenticate()
    from finite_reuse_replication_worker import files, publish
    producer, audit, plan = auth['producer'], auth['audit'], auth['plan']
    summary = {'version': VERSION, 'status': audit['advance']['status'], 'technical_complete': True,
        'registration': {'path': str(PLAN), **descriptor(PLAN)}, 'config': plan['config'],
        'rows': audit['rows'], 'baseline_rows': audit['baseline_rows'], 'prefix_rows': audit['prefix_rows'],
        'fits': producer['fits'], 'data_cases': audit['data_cases'], 'audit': audit,
        'regimes': producer['regimes'], 'oracle_metadata': producer['oracle_metadata'],
        'oracle_state_sha256': producer['oracle_state_sha256'],
        'producer_counts': producer['counts'], 'dataset_counts': producer['dataset_counts'],
        'structural_work_scope': producer['structural_work_scope'],
        'phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
        'generation_seconds': producer['generation_seconds'], 'prediction_times': producer['prediction_times'],
        'prefix_prediction_times': producer['prefix_prediction_times'], 'runtime': plan['runtime'],
        'qualification': auth['qualification'],
        'counts_scope': 'Producer counts describe learning and inference; audit.counts describe independent saved-output reconstruction only.',
        'replication_prerequisite': plan['replication_prerequisite'],
        'model_selection': False, 'architecture_claim': False, 'statistical_significance_claim': False}
    OUTPUT.mkdir()
    publish(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    report = document(summary)
    with (OUTPUT / 'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(finite-reuse-replication-results/benchmark.png)')
    overview = overview.replace('(../finite-reuse-replication-protocol.md)', '(finite-reuse-replication-protocol.md)')
    for name in ('summary.json', 'evidence.tar.gz', 'manifest.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(finite-reuse-replication-results/' + name + ')')
    with OVERVIEW.open('x') as stream:
        stream.write(overview)
    members = archive(OUTPUT, auth['study_files'])
    after = authenticate()
    require(after['study_files'] == auth['study_files'] and descriptor(Path(__file__).resolve()) == publisher_pin,
            'all original inputs and publisher unchanged after output')
    publish(OUTPUT / 'receipt.json', {'version': VERSION, 'status': 'PASS',
        'registration': summary['registration'], 'publisher': publisher_pin, 'files': files(OUTPUT),
        'overview': {'path': str(OVERVIEW), **descriptor(OVERVIEW)}, 'archive_members': members,
        'study_files': auth['study_files'], 'sources': plan['sources'],
        'phase_closures': {phase: {'receipt': descriptor(row['receipt_path']),
                                  'terminal': descriptor(row['terminal_path'])}
                           for phase, row in auth['phases'].items()},
        'replication_prerequisite': plan['replication_prerequisite'],
        'publication_counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls',
                                            'optimizer_calls', 'generator_calls', 'audit_reruns'), 0),
        'plotting_scope': 'Matplotlib uses saved JSON scalars after original closure; no checkpoint or scientific array is decoded.',
        'result_reads_after_original_closure': True, 'inputs_unchanged': True,
        'model_selection': False, 'visual_review_required': True})
    print(json.dumps({'output': str(OUTPUT), 'status': summary['status'], 'archive_members': members}))


if __name__ == '__main__':
    main()
