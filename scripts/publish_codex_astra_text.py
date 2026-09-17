"""Plot completed Codex-Astra aggregates without opening teacher packets or labels.

The frozen upstream report checks model and per-example provenance. This separate
publisher checks aggregate consistency and source hashes; it never loads models,
raw predictions, benchmark passages, gold labels, or protected evaluation packets.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'codex-astra-text-v1'
STEM = 'student-results'
ARMS = ('untrained', 'gold', 'astra_codex')
TASKS = ('boolq', 'clinc_domains')
SEEDS = [17, 29, 43]
COUNTS = {'boolq': 84, 'clinc_domains': 88}
CLASS_COUNTS = {'boolq': 2, 'clinc_domains': 11}
COLORS = ['#8290a6', '#268a77', '#6859bd']
LABELS = ['Untrained head\nreused control', 'Gold-label student\nreused control', 'Astra-label student\nnew training']
METRICS = ('accuracy', 'macro_class_accuracy', 'nll', 'multiclass_brier', 'median_latency_ms', 'p95_latency_ms')
PRODUCER_SOURCES = {
    'scripts/codex_astra_distillation.py', 'tests/test_codex_astra_distillation.py',
    *(f'src/openjev/research/{name}.py' for name in
      ('text_student', 'text_distillation', 'text_teacher', 'text_report')),
}
EXPECTED_PROTOCOL = {
    'version': VERSION, 'producer': 'codex_collaboration', 'requested_model': 'gpt-6-astra',
    'reasoning_effort': 'low', 'fork_turns': 'none', 'teacher_batches': 4, 'items_per_batch': 32,
    'teacher_items': 128, 'batch_shuffle_seed': 9172026, 'attempts_per_batch': 1,
    'fit_seeds': SEEDS, 'epochs': 3, 'learning_rate': 2e-5, 'weight_decay': .01, 'gradient_clip': 1.,
    'training_questions': 128, 'evaluation_questions': 172,
    'student': 'sentence-transformers/all-MiniLM-L6-v2',
    'student_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    'continuation': {'each_task_teacher_minus_untrained_accuracy_min': .10,
                     'each_task_teacher_minus_gold_accuracy_min': -.05},
    'direct_paid_api_calls': 0, 'independent_confirmation': False, 'novelty_established': False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def number(value, lower=0., upper=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < lower
            or (upper is not None and value > upper)):
        raise ValueError('Expected a finite aggregate within its valid range')
    return value


def close(actual, expected, description):
    if not math.isfinite(actual) or abs(actual - expected) > 1e-10:
        raise ValueError(f'Inconsistent {description}')


def check_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a lowercase SHA-256 digest')


def validate_aggregates(summary, provenance):
    protocol = summary['protocol']
    if (summary['status'] != 'completed' or any(protocol.get(key) != value for key, value in EXPECTED_PROTOCOL.items())
            or provenance['protocol'] != protocol or provenance['worker_packet_contents_published'] is not False
            or summary['independent_confirmation'] is not False or summary['novelty_established'] is not False
            or summary['direct_paid_api_calls'] != 0 or summary['codex_usage_cost'] != 'Unavailable'):
        raise ValueError('Expected the completed bounded Codex-Astra development report')
    for key in ('plan_sha256', 'source_packet_sha256'):
        check_hash(summary[key])
        if summary[key] != provenance[key]:
            raise ValueError('Aggregate and provenance source identity mismatch')
    check_hash(summary['teacher_completed_sha256'])
    expected_results = {f'{arm}/result-{seed}.json' for arm in ARMS for seed in SEEDS}
    if (set(summary['results_sha256']) != expected_results
            or summary['results_sha256'] != provenance['results_sha256']
            or set(provenance['sources_sha256']) != PRODUCER_SOURCES):
        raise ValueError('Expected exactly nine preserved result hashes and the frozen producer sources')
    for value in [*summary['results_sha256'].values(), *provenance['sources_sha256'].values()]:
        check_hash(value)
    if (set(summary['arms']) != set(ARMS) or set(summary['contrasts']) != set(TASKS)
            or set(summary['teacher_training_label_agreement']) != set(TASKS)):
        raise ValueError('Expected three arms and the two declared tasks')
    observed_classes = {}
    for arm in ARMS:
        record = summary['arms'][arm]
        if record['reused_control'] is not (arm != 'astra_codex') or set(record['tasks']) != set(TASKS):
            raise ValueError('Reused-control attribution or task coverage changed')
        number(record['training_seconds'])
        for task in TASKS:
            aggregates = record['tasks'][task]
            rows = aggregates['per_fit']
            if len(rows) != 3 or [row['seed'] for row in rows] != SEEDS:
                raise ValueError('Every arm and task requires all three ordered final fit seeds')
            for row in rows:
                if type(row['examples']) is not int or row['examples'] != COUNTS[task]:
                    raise ValueError('Unexpected reused development example count')
                for key in ('accuracy', 'macro_class_accuracy'):
                    number(row[key], upper=1.)
                number(row['nll'])
                number(row['multiclass_brier'], upper=2.)
                number(row['median_latency_ms'])
                number(row['p95_latency_ms'], lower=row['median_latency_ms'])
                classes = row['class_accuracy']
                if len(classes) != CLASS_COUNTS[task] or (task == 'boolq' and set(classes) != {'yes', 'no'}):
                    raise ValueError('Unexpected balanced task class coverage')
                if task in observed_classes and set(classes) != observed_classes[task]:
                    raise ValueError('Class identities differ across final fits')
                observed_classes[task] = set(classes)
                per_class = COUNTS[task] // CLASS_COUNTS[task]
                for value in classes.values():
                    number(value, upper=1.)
                    close(value * per_class, round(value * per_class), 'class-level correct count')
                close(row['accuracy'] * COUNTS[task], round(row['accuracy'] * COUNTS[task]), 'correct count')
                close(row['macro_class_accuracy'], np.mean(list(classes.values())), 'class-average accuracy')
                close(row['accuracy'], row['macro_class_accuracy'], 'balanced task accuracy')
            if set(aggregates['mean_across_fits']) != set(METRICS):
                raise ValueError('Unexpected across-fit metric coverage')
            for key in METRICS:
                close(aggregates['mean_across_fits'][key], np.mean([row[key] for row in rows]), f'mean {key}')
    expected_checks = {}
    for task in TASKS:
        agreement = number(summary['teacher_training_label_agreement'][task], upper=1.)
        close(agreement * 64, round(agreement * 64), '64-item teacher training agreement')
        contrasts = summary['contrasts'][task]
        if set(contrasts) != {'astra_codex_minus_untrained', 'astra_codex_minus_gold'}:
            raise ValueError('Expected the two declared student contrasts')
        for arm in ('untrained', 'gold'):
            contrast = contrasts[f'astra_codex_minus_{arm}']
            delta = number(contrast['delta'], lower=-1., upper=1.)
            means = summary['arms']
            expected_delta = (means['astra_codex']['tasks'][task]['mean_across_fits']['accuracy']
                              - means[arm]['tasks'][task]['mean_across_fits']['accuracy'])
            close(delta, expected_delta, 'student accuracy contrast')
            interval = contrast['exploratory_95_percent_interval']
            if (not isinstance(interval, list) or len(interval) != 2
                    or number(interval[0], lower=-1., upper=1.) > number(interval[1], lower=-1., upper=1.)):
                raise ValueError('Invalid exploratory interval')
        expected_checks[f'{task}_gain_vs_untrained'] = contrasts['astra_codex_minus_untrained']['delta'] >= .10
        expected_checks[f'{task}_within_gold_margin'] = contrasts['astra_codex_minus_gold']['delta'] >= -.05
    if (set(summary['checks']) != set(expected_checks)
            or any(type(value) is not bool for value in summary['checks'].values())
            or summary['checks'] != expected_checks or type(summary['continuation_passed']) is not bool
            or summary['continuation_passed'] != all(expected_checks.values())):
        raise ValueError('Continuation status does not follow the frozen thresholds and recorded contrasts')


def load_report(directory):
    summary, provenance = (strict_json(directory / name) for name in ('summary.json', 'provenance.json'))
    validate_aggregates(summary, provenance)
    for name, expected in provenance['sources_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen report producer source changed')
    return summary, provenance


def figure(summary, fixture=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.3), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, top=.785, bottom=.40, wspace=.18)
    title = 'SYNTHETIC TEST FIXTURE' if fixture else 'OpenJev: Astra-supervised text student'
    fig.suptitle(title, x=.075, y=.967, ha='left', fontsize=20, fontweight='bold', color='#243248')
    fig.text(.075, .917, '128 training questions  |  Three final fits  |  Reused public development evaluation',
             fontsize=11.5, color='#526174')
    handles = [Line2D([], [], color='#425066', marker=marker, linestyle='none', label=f'Fit seed {seed}',
                      markersize=6) for seed, marker in zip(SEEDS, ['o', '^', 's'], strict=True)]
    handles.append(Line2D([], [], color='#425066', linewidth=2, label='Mean of three fits'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.069, .882), frameon=False,
               ncol=4, fontsize=10, columnspacing=2.3)
    for ax, task, title in zip(axes, TASKS, ['BoolQ: 84 validation items', 'CLINC domains + OOS: 88 test items'], strict=True):
        for index, arm in enumerate(ARMS):
            values = [100 * row['accuracy'] for row in summary['arms'][arm]['tasks'][task]['per_fit']]
            for offset, value, marker in zip([-.12, 0, .12], values, ['o', '^', 's'], strict=True):
                ax.scatter(index + offset, value, marker=marker, s=67, c=COLORS[index],
                           linewidths=.9, edgecolors='white', zorder=4)
            average = float(np.mean(values))
            ax.plot([index - .24, index + .24], [average] * 2, color=COLORS[index], linewidth=2.5, zorder=3)
            label_y = min(101, max(values) + 5.5)
            ax.text(index, label_y, f'{average:.1f}%', ha='center', fontsize=11, color=COLORS[index], fontweight='bold')
        baseline = 100 / CLASS_COUNTS[task]
        ax.axhline(baseline, color='#a8afb8', linestyle=':', linewidth=1.2)
        ax.text(.02, .025, f'Dotted line: uniform choice ({baseline:.1f}%)',
                transform=ax.transAxes, fontsize=8, color='#6d7786',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 1})
        ax.set(xticks=range(3), xticklabels=LABELS, ylim=(-3, 107), xlim=(-.5, 2.5),
               yticks=[0, 25, 50, 75, 100])
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=11)
        ax.tick_params(axis='x', labelsize=10, length=0, pad=9)
        ax.tick_params(axis='y', labelsize=10)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_color('#c4cad2')
        ax.grid(axis='y', color='#e5e9ee', linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Student accuracy (%)', fontsize=11)
    fig.text(.075, .295, 'Teacher agreement with training labels', fontsize=11.2, fontweight='bold', color='#243248')
    for task, label, y in zip(TASKS, ['BoolQ', 'CLINC domains + OOS'], [.249, .208], strict=True):
        rate = summary['teacher_training_label_agreement'][task]
        fig.text(.075, y, label, fontsize=10.5, color='#425066')
        fig.text(.395, y, f'{rate * 100:.1f}% ({round(rate * 64)}/64)', ha='right', fontsize=11, color='#243248')
    fig.text(.075, .162, 'Training labels only; not teacher evaluation accuracy.', fontsize=9.1, color='#687486')
    passed = summary['continuation_passed']
    status, status_color = ('PASS', '#24725c') if passed else ('FAIL', '#9b462b')
    fig.text(.47, .295, f'Development continuation rule: {status}', fontsize=11.2,
             fontweight='bold', color=status_color)
    fig.text(.62, .258, 'vs untrained', fontsize=9.5, color='#687486')
    fig.text(.82, .258, 'vs gold', fontsize=9.5, color='#687486')
    for task, label, y in zip(TASKS, ['BoolQ', 'CLINC'], [.219, .178], strict=True):
        fig.text(.47, y, label, fontsize=10.5, color='#425066')
        for arm, key, x in [('untrained', 'gain_vs_untrained', .62), ('gold', 'within_gold_margin', .82)]:
            delta = summary['contrasts'][task][f'astra_codex_minus_{arm}']['delta'] * 100
            flag = summary['checks'][f'{task}_{key}']
            fig.text(x, y, f'{delta:+.1f} pp  {"PASS" if flag else "FAIL"}', fontsize=10.5,
                     color='#24725c' if flag else '#9b462b')
    fig.text(.47, .132, 'Each task needs at least +10 pp vs untrained and -5 pp vs gold.', fontsize=9.1, color='#687486')
    fig.text(.075, .076, 'Historical controls and evaluation items are reused. No independent confirmation or novelty claim.',
             fontsize=9.6, color='#526174')
    fig.text(.075, .041, 'Astra identity: explicit Codex worker model request, not provider-attested API metadata. Student scores are uncalibrated.',
             fontsize=9.1, color='#687486')
    return fig


def preserved_metadata(out):
    if not out.exists():
        return {}
    allowed = {'plan-metadata.json', 'provenance-audit.json', 'student-provenance-audit.json'}
    if not out.is_dir() or any(not path.is_file() or path.name not in allowed for path in out.iterdir()):
        raise FileExistsError('Output must be fresh or contain only the preserved plan metadata and provenance audits')
    return {path.name: sha(path) for path in sorted(out.iterdir())}


def publish(directory, out):
    summary, provenance = load_report(directory)
    previous = preserved_metadata(out)
    plt.rcParams.update({'svg.hashsalt': VERSION, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    fig = figure(summary)
    buffers = {}
    for suffix in ('png', 'svg'):
        stream = io.BytesIO()
        metadata = {'Software': STEM} if suffix == 'png' else {'Date': None, 'Creator': STEM}
        fig.savefig(stream, format=suffix, dpi=170, facecolor='white', metadata=metadata)
        data = stream.getvalue()
        if suffix == 'svg':
            data = ('\n'.join(line.rstrip() for line in data.decode().splitlines()) + '\n').encode()
        buffers[f'{STEM}.{suffix}'] = data
    plt.close(fig)
    plot_data = {key: summary[key] for key in ('protocol', 'plan_sha256', 'source_packet_sha256',
                                              'teacher_training_label_agreement', 'arms', 'contrasts', 'checks',
                                              'continuation_passed', 'independent_confirmation', 'novelty_established')}
    buffers['plot-data.json'] = (json.dumps(plot_data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    for name in ('summary.json', 'provenance.json'):
        buffers[name] = (directory / name).read_bytes()
    receipt = {'status': 'completed', 'protocol_version': VERSION,
               'summary_sha256': sha(directory / 'summary.json'), 'provenance_sha256': sha(directory / 'provenance.json'),
               'plan_sha256': summary['plan_sha256'], 'teacher_completed_sha256': summary['teacher_completed_sha256'],
               'publisher_sha256': sha(Path(__file__)), 'producer_sources_sha256': provenance['sources_sha256'],
               'preserved_metadata_sha256': previous,
               'result_files_sha256': summary['results_sha256'],
               'validation_scope': 'Aggregate consistency and frozen producer hashes; upstream report validates raw evidence',
               'models_loaded': False, 'raw_predictions_read': False, 'worker_packets_read': False,
               'development_items_per_fit': COUNTS, 'final_fit_seeds': SEEDS,
               'continuation_passed': summary['continuation_passed'], 'independent_confirmation': False,
               'novelty_established': False,
               'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in buffers.items()}}
    if preserved_metadata(out) != previous:
        raise ValueError('Preserved metadata changed while preparing figures')
    out.mkdir(parents=True, exist_ok=True)
    for name, data in buffers.items():
        with (out / name).open('xb') as stream:
            stream.write(data)
    with (out / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'out': str(out), 'continuation_passed': summary['continuation_passed']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    publish(args.report, args.out)


if __name__ == '__main__':
    main()
