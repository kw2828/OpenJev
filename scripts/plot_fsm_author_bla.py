"""Plot audited saved BLA/reference scalars, without arrays or model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import plot_fsm_linear_controls as reference_plot

REFERENCE_FAMILIES = ('varx96-ridge1e-06', 'tanh_output_only-lr0.001',
                      'tanh_feedback-lr0.0003')
LABELS = ('Author BLA28\n1 FIT-only fit', 'VARX96 / ridge 1e-6\n1 FIT-only fit',
          'Tanh output only\n3 frozen seeds', 'Tanh feedback\n3 unchanged seeds')
COLORS = ('#7554a3', '#64748b', '#287fa4', '#c56a25')
SCOPE = ('Exposed 100/200 mV DEV; C100/H128; 12 correlated-period records. '
         'Known input sequence, causal observed-output initialization; no future outputs. '
         'One author BLA28 refit versus unchanged references. No untouched-test or novelty claim.')
TIMING_SCOPE = ('Separate-process descriptive timings, not a controlled speed comparison. '
                'Full request including normalization, initialization and rollout; '
                'BLA includes a fresh observability solve. One warm-up and 24 saved requests per instance. '
                'Reference neural points are seed medians; diamonds are their mean.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def read_json(path):
    def invalid(token):
        raise ValueError('nonfinite JSON: ' + token)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def extract(summary, evaluation, reference_summary, reference_evaluations):
    """Only aggregate saved scalars; never regenerate predictions or select a model."""
    require(summary.get('status') in ('REFERENCE_COMPLETE', 'REFERENCE_INCOMPLETE'), 'BLA summary status')
    indexed = reference_plot.records(evaluation.get('rows'))
    require(set(indexed) == reference_plot.RECORDS, 'exact BLA record roster required')
    complete = all(row['status'] == 'complete' for row in indexed.values())
    mean = summary.get('mean_rmse')
    if complete:
        require(finite(mean) and math.isclose(mean, fmean(row['rmse'] for row in indexed.values()),
                                            rel_tol=1e-10, abs_tol=1e-12), 'BLA saved mean mismatch')
    else:
        require(mean is None, 'incomplete BLA cannot have a complete mean')
    timing = evaluation.get('median_request_ms')
    if timing is not None:
        samples = evaluation.get('request_ms')
        require(evaluation.get('timing_error') is None and finite(timing, positive=True)
                and isinstance(samples, list) and len(samples) == 24
                and all(finite(value, positive=True) for value in samples), 'invalid BLA timing')
    require(summary.get('median_request_ms') == timing, 'BLA summary/evaluation timing mismatch')
    prior = reference_plot.extract(reference_summary, reference_evaluations)
    prior_by_family = {row['family']: row for row in prior['families']}
    rows = [{'family': 'author_bla28', 'label': LABELS[0], 'status': summary['status'],
             'mean_rmse': mean, 'mean_seed_median_ms': timing,
             'members': [{'seed': None, 'rmse': mean, 'latency_ms': timing}],
             'persistent_numeric_bytes': summary.get('persistent_numeric_bytes'),
             'complete_records': sum(row['status'] == 'complete' for row in indexed.values()),
             'fit_status': summary.get('fit', {}).get('status')}]
    for index, family in enumerate(REFERENCE_FAMILIES, 1):
        saved = prior_by_family[family]
        rows.append({'family': family, 'label': LABELS[index],
                     'status': 'complete' if saved['eligible'] else 'incomplete',
                     'mean_rmse': saved['mean_rmse'],
                     'mean_seed_median_ms': saved['mean_seed_median_ms'],
                     'members': [{'seed': member['seed'], 'rmse': member['rmse'],
                                  'latency_ms': member['latency_ms']} for member in saved['members']],
                     'persistent_numeric_bytes': saved['persistent_numeric_bytes']})
    return {'families': rows, 'scope': SCOPE, 'timing_scope': TIMING_SCOPE,
            'aggregation': 'Arithmetic mean of 12 record RMSEs per instance, then mean over neural seeds. '
                           'Seed dots are not confidence intervals; periods are correlated.',
            'saved_bla_summary': summary, 'saved_bla_evaluation': evaluation,
            'saved_reference_summary': reference_summary,
            'saved_reference_evaluations': reference_evaluations,
            'no_model_or_prediction_array_access': True}


def figure(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), sharey=True)
    positions = list(range(4))
    for axis, mean_key, point_key, title, xlabel in (
        (axes[0], 'mean_rmse', 'rmse', 'Exposed DEV forecast error', 'Mean standardized RMSE (lower is better)'),
        (axes[1], 'mean_seed_median_ms', 'latency_ms', 'Separate-process timing: descriptive only',
         'Full-request latency, ms (not a controlled speed comparison)'),
    ):
        numbers = []
        for position, row, color in zip(positions, values['families'], COLORS, strict=True):
            members = row['members']
            for index, member in enumerate(members):
                value = member[point_key]
                offset = (index-(len(members)-1)/2)*.13
                if value is None:
                    axis.text(.01, position+offset, 'Missing/failed', transform=axis.get_yaxis_transform(),
                              fontsize=8, va='center', color='#a22')
                else:
                    numbers.append(value)
                    axis.scatter(value, position+offset, s=32, facecolors='none', edgecolors=color,
                                 linewidths=1.4, zorder=3, clip_on=False)
            mean = row[mean_key]
            if mean is not None:
                numbers.append(mean)
                axis.scatter(mean, position, s=68, marker='D', color=color, zorder=4, clip_on=False)
                axis.annotate(f'{mean:.4g}', (mean, position), xytext=(8, 0),
                              textcoords='offset points', va='center', fontsize=10)
            if row['status'] not in ('complete', 'REFERENCE_COMPLETE'):
                axis.text(.99, position+.25, row['status'].replace('_', ' '),
                          transform=axis.get_yaxis_transform(), ha='right', va='center', fontsize=8, color='#a22')
        high = max(numbers, default=1.)
        axis.set_xlim(0, high*1.30 if high > 0 else 1.)
        axis.set_ylim(3.55, -.55)
        axis.set_title(title, pad=14)
        axis.set_xlabel(xlabel, labelpad=10)
        axis.grid(axis='x', color='#e2e8f0')
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(positions, LABELS)
    axes[0].tick_params(axis='y', length=0, pad=10)
    fig.suptitle('Author BLA28 and retained FSM references', fontsize=16, y=.97)
    fig.legend(handles=[Line2D([], [], color='#334155', marker='D', linestyle='none', label='Family mean'),
                        Line2D([], [], color='#334155', marker='o', markerfacecolor='none', linestyle='none',
                               label='One fit or individual seed')],
               loc='upper center', bbox_to_anchor=(.56, .91), ncol=2, frameon=False)
    fig.text(.02, .072, 'C100/H128, exposed 100/200 mV DEV; 12 correlated-period records. '
             'All reference weights unchanged; BLA refitted on FIT only.', fontsize=9)
    fig.text(.02, .035, 'Known input sequence; no future outputs. No untouched-test, novelty or controlled speed claim. '
             'Neural timing: mean of three seed medians.', fontsize=9)
    fig.subplots_adjust(left=.19, right=.975, top=.78, bottom=.23, wspace=.27)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'benchmark.{extension}', dpi=170)
    plt.close(fig)


def authenticate(study, audit_path, audit_process, reference_study):
    """Bind saved scalar files to the independent audit before reading their values."""
    audit_pin = descriptor(audit_path)
    closure = read_json(audit_process)
    require(closure.get('observed_exit_code') == 0 and closure.get('audit_sha256') == audit_pin['sha256'],
            'original audit closure/output join required')
    command = closure.get('command')
    require(isinstance(command, list) and '--output' in command
            and Path(command[command.index('--output')+1]).resolve() == audit_path,
            'original audit command/output mismatch')
    audit = read_json(audit_path)
    require(audit.get('status') == 'PASS' and audit.get('agreement') is True, 'closed PASS audit required')
    require(audit.get('study') == str(study), 'audited study path mismatch')
    admission = audit.get('inputs')
    require(isinstance(admission, dict), 'audit input pins required')
    files = admission.get('files')
    require(isinstance(files, dict), 'audited study file roster required')
    paths = {'bla_summary': study / 'summary.json', 'bla_evaluation': study / 'evaluation.json',
             'reference_summary': reference_study / 'summary.json',
             'reference_evaluations': reference_study / 'evaluations.json'}
    expected = {key: {'path': str(path), **files[path.name]}
                for key, path in paths.items() if key.startswith('bla_')}
    expected.update({key: admission.get(key) for key in ('reference_summary', 'reference_evaluations')})
    require(all(descriptor(path) == expected[key] for key, path in paths.items()), 'audit scalar input pin mismatch')
    process = admission.get('process')
    require(isinstance(process, dict) and set(process) == {'path', 'bytes', 'sha256'}, 'original process pin required')
    require(descriptor(process['path']) == process, 'original process changed')
    closed = read_json(process['path'])
    require(closed.get('status') == 'completed' and closed.get('observed_exit_code') == 0
            and closed.get('end_identity_matches') is True, 'original study did not close successfully')
    require(closed.get('summary.json_sha256') == expected['bla_summary']['sha256'], 'process summary join')
    producer_command = closed.get('command')
    require(isinstance(producer_command, list) and '--output' in producer_command
            and Path(producer_command[producer_command.index('--output')+1]).resolve() == study,
            'original study command/output mismatch')
    return paths, {'audit': audit_pin, 'audit_process': descriptor(audit_process),
                   'original_process': process, **expected}


def run(study, audit, audit_process, reference_study, output):
    study, audit, audit_process, reference_study, output = (
        Path(path).resolve() for path in (study, audit, audit_process, reference_study, output))
    require(not output.exists(), 'exclusive output required')
    require(not output.is_relative_to(study) and not output.is_relative_to(reference_study)
            and not output.is_relative_to(audit.parent), 'plot output must not alter admitted evidence roots')
    paths, inputs = authenticate(study, audit, audit_process, reference_study)
    sources = {str(path): descriptor(path) for path in (Path(__file__).resolve(), Path(reference_plot.__file__).resolve())}
    values = extract(*(read_json(paths[key]) for key in
                       ('bla_summary', 'bla_evaluation', 'reference_summary', 'reference_evaluations')))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plotted-values.json', values)
    figure(values, output)
    require(all(descriptor(value['path']) == value for value in (*inputs.values(), *sources.values())),
            'input or renderer changed during rendering')
    outputs = {name: descriptor(output / name) for name in ('benchmark.png', 'benchmark.pdf', 'plotted-values.json')}
    write_json(output / 'receipt.json', {'study': str(study), 'reference_study': str(reference_study),
                                       'inputs': inputs, 'sources': sources, 'outputs': outputs,
                                       'scope': SCOPE, 'timing_scope': TIMING_SCOPE})
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--audit-process', required=True, type=Path)
    parser.add_argument('--reference-study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.audit, args.audit_process, args.reference_study, args.output), indent=2))
