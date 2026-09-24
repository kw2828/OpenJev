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
STUDY = ROOT / 'output/finite-reuse-learning-v1'
PLAN = STUDY / 'study-registration.json'
PLAN_SHA = '3fed1607e9f2ad572ba2ba0f5c90225ff7e98b74ab6a0846f1e76e5d3d7d46de'
ENGINEERING_SHA = '1a938b4b4f798960fb6a07ef7880f24db165b950824df8e79cfc07e8531ca0d3'
OUTPUT = ROOT / 'research/finite-reuse-learning-results'
OVERVIEW = ROOT / 'research/finite-reuse-learning-results.md'
VERSION = 'finite-reuse-learning-publication-v1'
ARMS = ('original_free', 'matched_free', 'rounded')
SEEDS = (434261001, 434261002, 434261003)
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
    require(len(plan['sources']) == 102 and all(descriptor(ROOT / name) == expected
            for name, expected in plan['sources'].items()), 'all frozen sources before helper imports')
    from finite_reuse_learning_worker import (
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
    require(phases['fit']['receipt']['result'] == {'fits': 9, 'rows': 36}, 'all nine fits and endpoint rows')
    study_files = files(STUDY)
    # All current scientific results are first read after the checks above.
    producer = read(Path(plan['phases']['fit']['output']) / 'summary.json')
    audit = read(Path(plan['phases']['audit']['output']) / 'audit.json')
    from audit_finite_observation_learning import compare
    require(producer['version'] == 'finite-reuse-learning-v1' and producer['implementation'] == 'reuse'
            and producer['config'] == plan['config'] and audit['version'] == 'finite-reuse-learning-audit-v1'
            and audit['profile'] == 'science' and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['exact_oracle_agreement'] is True, 'original independently audited scientific record')
    expected_fits = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(producer['fits']) == 9 and {(r['arm'], r['seed']) for r in producer['fits']} == expected_fits,
            'complete fixed fit roster')
    require(len(audit['rows']) == 36 and {(r['arm'], r['seed'], r['horizon']) for r in audit['rows']}
            == {(arm, seed, h) for arm, seed in expected_fits for h in (1, 2, 4, 8)}, 'complete metric roster')
    row_key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(producer['rows'], key=row_key), sorted(audit['rows'], key=row_key))
    compare(sorted(producer['baseline_rows'], key=lambda r: r['horizon']), audit['baseline_rows'])
    compare(producer['prefix_rows'], audit['prefix_rows'])
    require(len(audit['prefix_rows']) == 9 and producer['structural_work'] == audit['structural_work'],
            'all prefix rows and disjoint actual work')
    require(audit['metadata']['fits'] == 9 and audit['metadata']['payloads'] == 88
            and audit['counts'] == {'array_decodes': 45, 'checkpoint_decodes': 27,
                'optimizer_json_decodes': 27, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0}, 'complete saved-output audit scope')
    require(set(audit['gates']) == set(ARMS), 'all three arm criteria')
    for arm in ARMS:
        require(set(audit['gates'][arm]) == set(CRITERIA), 'all three absolute criteria')
        for gate in audit['gates'][arm].values():
            require(all(type(v) is bool for v in gate['conditions'].values())
                    and gate['passed'] == all(gate['conditions'].values()), 'saved absolute gate conjunction')
    advance = audit['advance']
    require(len(advance['conditions']) == 19 and all(type(v) is bool for v in advance['conditions'].values())
            and advance['passed'] == all(advance['conditions'].values())
            and advance['status'] == 'UPDATE_MATCHED_ADVANCE_' + ('PASS' if advance['passed'] else 'FAIL')
            and audit['allocation_comparisons']['advance'] == advance
            and len(audit['allocation_comparisons']['paired']) == 12
            and len(audit['allocation_comparisons']['mean_regret']) == 4
            and len(audit['allocation_comparisons']['mean_fit_time']) == 2
            and phases['audit']['receipt']['result'] == {'gates': audit['gates'], 'advance': advance},
            'all original nineteen continuation conditions and saved audit receipt')
    require(producer['counts']['fit_count'] == 9 and producer['counts']['accepted_prefix_steps'] == 9216
            and producer['counts']['accepted_joint_steps'] == 27648
            and producer['counts']['accepted_optimizer_steps'] == 36864
            and producer['checkpoint_barrier']['dev_generation_count'] == 0,
            'exact scientific update counts and pre-DEV barrier')
    log = (Path(plan['phases']['qualify']['output']) / 'command-1.log').read_text()
    require(re.findall(r'(?m)^\s*(\d+) passed, (\d+) warning in [^\n]+$', log) == [('229', '1')],
            'original selected qualification result')
    return {'plan': plan, 'phases': phases, 'producer': producer, 'audit': audit, 'study_files': study_files}


def chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), layout='constrained')
    colors = ('#2864a3', '#d17721', '#248372')
    labels = ('Original\nfree', 'Matched\nfree', 'Rounded')
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in summary['rows']}
    fitted = {(r['arm'], r['seed']): r for r in summary['fits']}
    for panel, axis in enumerate(axes):
        horizon = (4, 8)[panel] if panel < 2 else None
        for j, seed in enumerate(SEEDS):
            ys = [keyed[arm, seed, horizon]['blind_regret'] if horizon is not None
                  else fitted[arm, seed]['seconds'] for arm in ARMS]
            xs = [i + (j - 1) * .065 for i in range(3)]
            axis.plot(xs, ys, color=colors[j], marker='o', linewidth=1, markersize=5,
                      alpha=.85, label=str(seed))
        for i, arm in enumerate(ARMS):
            values = [keyed[arm, seed, horizon]['blind_regret'] if horizon is not None
                      else fitted[arm, seed]['seconds'] for seed in SEEDS]
            axis.scatter(i, math.fsum(values) / 3, color='#152130', marker='D', s=35,
                         zorder=4, label='Mean' if i == 0 else None)
        axis.set_xticks(range(3), labels)
        axis.set_xlim(-.35, 2.35)
        if horizon is not None and all(keyed[arm, seed, horizon]['blind_regret'] > 0
                                       for arm in ARMS for seed in SEEDS):
            axis.set_yscale('log')
            axis.set_ylabel('Regret, log scale (lower is better)')
        else:
            axis.set_ylim(bottom=0)
            axis.set_ylabel('Regret (lower is better)' if horizon else 'Seconds (lower is faster)')
        axis.set_title(f'H{horizon} blind decision regret' if horizon else 'Complete fit time', fontsize=12)
        axis.grid(axis='y', color='#e4e8ed')
        axis.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Fixed update counts: all three seeds and every model', fontsize=14, weight='bold')
    fig.savefig(path, dpi=180, metadata={'Description':
        'Saved independently audited JSON only. All nine fits; paired lines identify fit seeds. Means have no confidence interval. No model or checkpoint decoding.'})
    plt.close(fig)


def document(summary):
    audit = summary['audit']
    lines = ['# Equal updates with shared computation', '',
        f"**{summary['status']}**. {sum(audit['advance']['conditions'].values())}/19 prospectively fixed continuation conditions pass.", '',
        ('All nine fits completed 1,024 prefix updates and 3,072 joint updates each. '
        'Same-seed arms received identical joint minibatches. All use the qualified '
        'per-call reuse implementation. Equal update counts do not imply equal runtime, '
        'FLOPs, gradients or effective degrees of freedom.'), '',
        '![All seeds: H4/H8 blind regret and full fit time](benchmark.png)', '',
        '| Model | Short horizon | Blind extrapolation | Observed filtering |',
        '|---|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS, strict=True):
        values = []
        for criterion in CRITERIA:
            gate = audit['gates'][arm][criterion]
            values.append(f"{'PASS' if gate['passed'] else 'FAIL'} {sum(gate['conditions'].values())}/{len(gate['conditions'])}")
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines += ['', 'All seeds remain binding. Means do not rescue a failed absolute condition or a worse paired regret.', '',
        '| Control | Horizon | Rounded mean regret | Control mean regret | Relative reduction |',
        '|---|---:|---:|---:|---:|']
    for row in audit['allocation_comparisons']['mean_regret']:
        reduction = 'undefined (zero control)' if row['relative_reduction'] is None else f"{100 * row['relative_reduction']:.2f}%"
        lines.append(f"| {row['control']} | {row['horizon']} | {row['candidate_mean_regret']:.6g} | {row['control_mean_regret']:.6g} | {reduction} |")
    lines += ['', ('Relative reductions are ratios of the three fit means, with positive values favoring rounded. '
        'These descriptive summaries are not significance tests. The very large mean reductions are driven by '
        'weak control fits at seeds 434261002 and 434261003; the first seed of each control performs much better. '
        'All twelve paired comparisons still favor rounded. The regret panels use a log scale to retain these '
        'differences; black diamonds are arithmetic means.'), '',
        '| Control | Rounded mean fit seconds | Control mean fit seconds | Extra rounded time |',
        '|---|---:|---:|---:|']
    for row in audit['allocation_comparisons']['mean_fit_time']:
        lines.append(f"| {row['control']} | {row['candidate_mean_seconds']:.6f} | {row['control_mean_seconds']:.6f} | {100 * (row['ratio'] - 1):.2f}% |")
    lines += ['', '| Model | Fit seed | Prefix updates | Joint updates | Prefix stage seconds | Joint stage seconds | Complete fit seconds | Controller seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    stages = {(r['arm'], r['seed']): r['stages'] for r in audit['allocation']['stages']}
    for row in sorted(summary['fits'], key=lambda r: (ARMS.index(r['arm']), r['seed'])):
        stage_seconds = [s['stopped_elapsed'] - s['start_elapsed'] for s in stages[row['arm'], row['seed']]]
        lines.append(f"| {row['arm']} | {row['seed']} | {row['accepted_prefix_updates']} | {row['updates']} | {stage_seconds[0]:.6f} | {stage_seconds[1]:.6f} | {row['seconds']:.6f} | {row['timed_seconds']:.6f} |")
    times = summary['phase_seconds']
    lines += ['', (f"Original native phases: qualification {times['qualify']:.6f}s, fit {times['fit']:.6f}s, audit {times['audit']:.6f}s. "
        'The selected qualification ran 229 passing tests with one warning and retained the independently audited exposure probe.'), '',
        ('Complete fit time includes setup, updates, guards, boundary checkpoints and allocation-log persistence. '
        'Controller time includes its final summary and is nested inside full fit time. Generation, inference, fit and audit '
        'durations must not be added again to their enclosing phase. Runtime comparisons are descriptive, not continuation conditions. '
        'Prefix and joint stage durations are also nested within the controller; setup, boundary and summary work falls outside these stage intervals.'), '',
        (f"Fresh base-condition development retains {summary['data_cases']['base']} of 128 attempted prefixes; "
        f"TRAIN retains {summary['data_cases']['train']} of 512. All attempted prefixes, including found events, contribute "
        'to prefix likelihood; endpoint metrics use retained prefixes. Training endpoint labels cover H1/H2, while H4/H8 '
        'are extrapolation. Learned models receive public histories, not the exact reference state. The synthetic world '
        'uses a known doubly stochastic transition prior, and all arms have privileged starting readouts. '
        'These choices favor the tested model family. This is no claim of native transfer, latent identification, '
        'architectural novelty or statistical significance.'), '',
        ('The independent audit reconstructs saved targets, schedules, all 27 model and 27 Adam boundaries, and actual '
        'forward work. It does not replay training. The publication reads JSON and hashes opaque evidence; zero learner '
        'calls in the publication or audit do not describe the training phase. The five joint reuse routes count physical '
        'shared operations once, separately from logical rollout counts and unmeasured backward FLOPs.'), '',
        ('The earlier equal-time rounded study did not meet its advancement criterion. This fresh equal-update '
        'result does not overturn that result or establish equal-time superiority: rounded takes more time here. '
        'The implementation, optimization exposure and fresh cohort also differ across studies, so their contrast '
        'does not isolate one causal explanation.'), '',
        ('The next step is an untouched replication with a predeclared scenario shift, followed by a second '
        'environment. A local pass is not ICLR readiness, calibrated probabilities or evidence of biological '
        'benefit. These follow-ups require separate frozen protocols; this report does not execute or admit them.'), '',
        'All 19 continuation conditions:', '', '| Condition | Result |', '|---|---|']
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in audit['advance']['conditions'].items())
    lines += ['', 'Every failed absolute condition:', '']
    failures = [f"- {arm} / {criterion}: {name}" for arm in ARMS for criterion in CRITERIA
                for name, value in audit['gates'][arm][criterion]['conditions'].items() if not value]
    lines.extend(failures or ['None.'])
    lines += ['', 'All endpoint results:', '',
        '| Model | Seed | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE | Observed survival MAE | Shuffled regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl',
               'blind_survival_mae', 'observed_survival_mae', 'shuffled_blind_regret')
    for row in sorted(summary['rows'], key=lambda r: (ARMS.index(r['arm']), r['seed'], r['horizon'])):
        lines.append(f"| {row['arm']} | {row['seed']} | {row['horizon']} | " +
                     ' | '.join(f'{row[key]:.8g}' for key in metrics) + ' |')
    lines += ['', ('[Protocol](../finite-reuse-learning-protocol.md) · [Complete saved summary](summary.json) · '
        '[Current-study evidence](evidence.tar.gz) · [Manifest](manifest.json) · [Publication receipt](receipt.json)'), '',
        ('The archive contains this complete current study, both 102-source snapshots, the engineering exposure probe, '
        'original closures, every saved model/optimizer boundary and publication artifacts. Earlier linked throughput '
        'and stopped studies, the interpreter and installed packages remain external. Their registered prerequisite '
        'evidence is reauthenticated, and its original descriptors are retained. Prior stopped attempts remain closed.'), '']
    return '\n'.join(lines)


def archive(output, study_files):
    from finite_reuse_learning_worker import publish
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
    from finite_reuse_learning_worker import files, publish
    producer, audit, plan = auth['producer'], auth['audit'], auth['plan']
    summary = {'version': VERSION, 'status': audit['advance']['status'], 'technical_complete': True,
        'registration': {'path': str(PLAN), **descriptor(PLAN)}, 'config': plan['config'],
        'rows': audit['rows'], 'baseline_rows': audit['baseline_rows'], 'prefix_rows': audit['prefix_rows'],
        'fits': producer['fits'], 'data_cases': audit['data_cases'], 'audit': audit,
        'producer_counts': producer['counts'], 'dataset_counts': producer['dataset_counts'],
        'structural_work_scope': producer['structural_work_scope'],
        'phase_seconds': {k: v['terminal']['wall_seconds'] for k, v in auth['phases'].items()},
        'generation_seconds': producer['generation_seconds'], 'prediction_times': producer['prediction_times'],
        'prefix_prediction_times': producer['prefix_prediction_times'], 'runtime': plan['runtime'],
        'qualification_tests_passed': 229, 'qualification_warnings': 1,
        'counts_scope': 'Producer counts describe learning and inference; audit.counts describe independent saved-output reconstruction only.',
        'throughput_qualification': plan['throughput_qualification'],
        'model_selection': False, 'architecture_claim': False, 'statistical_significance_claim': False}
    OUTPUT.mkdir()
    publish(OUTPUT / 'summary.json', summary)
    chart(summary, OUTPUT / 'benchmark.png')
    report = document(summary)
    with (OUTPUT / 'report.md').open('x') as stream:
        stream.write(report)
    overview = report.replace('(benchmark.png)', '(finite-reuse-learning-results/benchmark.png)')
    overview = overview.replace('(../finite-reuse-learning-protocol.md)', '(finite-reuse-learning-protocol.md)')
    for name in ('summary.json', 'evidence.tar.gz', 'manifest.json', 'receipt.json'):
        overview = overview.replace('(' + name + ')', '(finite-reuse-learning-results/' + name + ')')
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
        'throughput_qualification': plan['throughput_qualification'],
        'publication_counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls',
                                            'optimizer_calls', 'generator_calls', 'audit_reruns'), 0),
        'plotting_scope': 'Matplotlib uses saved JSON scalars after original closure; no checkpoint or scientific array is decoded.',
        'result_reads_after_original_closure': True, 'inputs_unchanged': True,
        'model_selection': False, 'visual_review_required': True})
    print(json.dumps({'output': str(OUTPUT), 'status': summary['status'], 'archive_members': members}))


if __name__ == '__main__':
    main()
