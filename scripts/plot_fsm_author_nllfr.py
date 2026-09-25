"""Render closed, independently audited FSM scalars; never decode numeric arrays."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean, median

import audit_fsm_author_nllfr as admission
import plot_fsm_author_bla as bla_plot

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = 'scripts/audit_fsm_author_nllfr.py'
AUDITOR_TEST = 'tests/test_audit_fsm_author_nllfr.py'
AUDITOR_SHA = {AUDITOR: '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
               AUDITOR_TEST: '2c824eed78bb21f4d1f709864bc31cdf9042955dfee4e5754c9c2eb54a4e5951'}
AUDITOR_FREEZE = 'output/fsm-author-engineering-v1/nllfr-auditor-freeze.json'
COLORS = ('#b44d55', *bla_plot.COLORS)
SCOPE = ('Exposed 100/200 mV DEV, C100/H128, 12 correlated-period records. '
         'Known input sequence with causal observed-output initialization; no future outputs. '
         'NL-LFR scores from an incomplete fit are diagnostic only. '
         'No candidate-survival, untouched-test, control or novelty claim.')
TIMING_SCOPE = ('Separate-process descriptive full-request timings, not a controlled speed comparison. '
                'Normalization, initialization and rollout included. NL-LFR includes a bounded nonlinear '
                'context solve; BLA includes a fresh observability solve. One warm-up and 24 requests '
                'per instance. Neural family diamonds average three saved seed medians.')
require, descriptor = bla_plot.require, bla_plot.descriptor
read_json, write_json, finite = bla_plot.read_json, bla_plot.write_json, bla_plot.finite


def extract(summary, evaluation, bla_summary, bla_evaluation, reference_summary, reference_evaluations):
    """Check saved scalar arithmetic, preserving incomplete fits and missing costs."""
    require(summary.get('status') in ('REFERENCE_COMPLETE', 'REFERENCE_INCOMPLETE'), 'NL reference status')
    require(summary.get('fit_status') in ('complete', 'iteration_cap_reached'), 'NL fit status')
    rows = evaluation.get('rows')
    require(isinstance(rows, list) and len(rows) == 12, 'twelve NL record slots required')
    indexed = {}
    for row in rows:
        name = row.get('record_id')
        require(name in bla_plot.reference_plot.RECORDS and name not in indexed, 'NL record identity')
        require(row.get('status') in ('complete', 'incomplete'), 'NL record status')
        if row['status'] == 'complete':
            require(finite(row.get('rmse')) and row.get('requests') == 32 and row.get('horizon') == 128,
                    'NL complete record geometry/value')
        else:
            require(type(row.get('completed_requests')) is int and 0 <= row['completed_requests'] < 32
                    and row.get('expected_requests') == 32, 'NL incomplete record geometry')
        indexed[name] = row
    complete = all(r['status'] == 'complete' for r in rows)
    mean = summary.get('mean_rmse')
    require((finite(mean) and math.isclose(mean, fmean(r['rmse'] for r in rows), rel_tol=1e-10, abs_tol=1e-12))
            if complete else mean is None, 'NL saved complete mean mismatch')
    timing, samples = evaluation.get('median_request_ms'), evaluation.get('timings')
    require(isinstance(samples, list) and len(samples) <= 24, 'NL timing slots')
    for sample in samples:
        require(sample.get('record_id') in indexed and sample.get('start') in (0, 7936)
                and finite(sample.get('request_ms'), positive=True), 'NL timing identity/value')
    require(len({(r['record_id'], r['start']) for r in samples}) == len(samples), 'duplicate NL timing')
    if timing is not None:
        require(len(samples) == 24 and evaluation.get('timing_error') is None and finite(timing, positive=True)
                and math.isclose(timing, median(s['request_ms'] for s in samples), rel_tol=1e-10, abs_tol=1e-12),
                'NL complete timing mismatch')
    else:
        require(isinstance(evaluation.get('timing_error'), str) and bool(evaluation['timing_error']),
                'NL unavailable timing requires reason')
    require(summary.get('median_request_ms') == timing, 'NL summary timing mismatch')
    require(type(summary.get('persistent_numeric_bytes')) is int and summary['persistent_numeric_bytes'] > 0
            and summary['persistent_numeric_bytes'] == evaluation.get('persistent_numeric_bytes'), 'NL storage mismatch')
    expected = 'REFERENCE_COMPLETE' if summary['fit_status'] == 'complete' and complete and timing is not None else 'REFERENCE_INCOMPLETE'
    require(summary['status'] == expected, 'incomplete fit cannot become complete through finite forecasts')
    prior = bla_plot.extract(bla_summary, bla_evaluation, reference_summary, reference_evaluations)
    label = 'NL-LFR28 diagnostic\nFIT INCOMPLETE' if summary['fit_status'] != 'complete' else 'Author NL-LFR28\n1 FIT-only fit'
    first = {'family': 'author_nllfr28', 'label': label, 'status': summary['status'],
             'fit_status': summary['fit_status'], 'mean_rmse': mean, 'mean_seed_median_ms': timing,
             'complete_records': sum(r['status'] == 'complete' for r in rows),
             'persistent_numeric_bytes': summary['persistent_numeric_bytes'],
             'members': [{'seed': None, 'rmse': mean, 'latency_ms': timing}],
             'timing_error': evaluation.get('timing_error')}
    return {'families': [first, *prior['families']], 'scientific_status': summary['status'],
            'fit_status': summary['fit_status'], 'scope': SCOPE, 'timing_scope': TIMING_SCOPE,
            'aggregation': prior['aggregation'], 'no_model_or_prediction_array_access': True,
            'saved_nllfr_summary': summary, 'saved_nllfr_evaluation': evaluation,
            **{k: v for k, v in prior.items() if k.startswith('saved_')}}


def figure(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, field, point, title, xlabel in (
        (axes[0], 'mean_rmse', 'rmse', 'Exposed DEV forecast error', 'Mean standardized RMSE (lower is better)'),
        (axes[1], 'mean_seed_median_ms', 'latency_ms', 'Full-request latency: descriptive only', 'Latency, ms; separate processes'),
    ):
        numbers = []
        for i, (row, color) in enumerate(zip(values['families'], COLORS, strict=True)):
            for j, member in enumerate(row['members']):
                number = member[point]
                offset = (j-(len(row['members'])-1)/2)*.13
                if number is None:
                    ax.text(.01, i+offset, 'Missing/failed', transform=ax.get_yaxis_transform(), color='#a22', fontsize=8)
                else:
                    numbers.append(number)
                    ax.scatter(number, i+offset, facecolors='none', edgecolors=color, s=32, clip_on=False, zorder=3)
            number = row[field]
            if number is not None:
                numbers.append(number)
                ax.scatter(number, i, color=color, marker='D', s=62, clip_on=False, zorder=4)
                ax.annotate(f'{number:.4g}', (number, i), xytext=(8, 0), textcoords='offset points', va='center')
            if row['status'] not in ('complete', 'REFERENCE_COMPLETE'):
                ax.text(.99, i+.29, row['status'].replace('_', ' '), transform=ax.get_yaxis_transform(),
                        ha='right', color='#a22', fontsize=8)
        spec = bla_plot.reference_plot.axis_spec(numbers)
        if spec['scale'] == 'symlog':
            ax.set_xscale('symlog', linthresh=spec['linthresh'])
            xlabel += ' (symlog; zero included)'
        ax.set_xlim(*spec['limits'])
        ax.set_ylim(4.6, -.6)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=15)
        ax.grid(axis='x', color='#e2e8f0')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(5), [r['label'] for r in values['families']])
    axes[0].tick_params(axis='y', length=0, pad=10)
    incomplete = values['scientific_status'] != 'REFERENCE_COMPLETE'
    fig.suptitle('NL-LFR reference incomplete: diagnostic forecasts only' if incomplete else
                 'NL-LFR28 and retained FSM references', fontsize=16, y=.97)
    fig.legend(handles=[Line2D([], [], color='#334155', marker='D', linestyle='none', label='Family mean'),
                        Line2D([], [], color='#334155', marker='o', markerfacecolor='none', linestyle='none',
                               label='One fit or individual seed')],
               loc='upper center', bbox_to_anchor=(.58, .91), ncol=2, frameon=False)
    fig.text(.02, .113, 'C100/H128, exposed 100/200 mV DEV; 12 correlated-period records. Known inputs; no future outputs.', fontsize=9)
    fig.text(.02, .08, 'NL-LFR includes a bounded nonlinear context solve; BLA a fresh linear state solve. '
             'All retained reference weights unchanged.', fontsize=9)
    fig.text(.02, .047, 'Separate-process latency is descriptive. No candidate-survival, untouched-test, '
             'novelty or controlled speed claim.', fontsize=9)
    fig.subplots_adjust(left=.21, right=.975, top=.78, bottom=.24, wspace=.29)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'benchmark.{extension}', dpi=170)
    plt.close(fig)


def authenticate(study, audit, audit_process, evaluation, reference, bla):
    """Return (six scalar paths, flat file descriptors); no numeric array access.

    The qualified auditor's metadata-only admission rechecks the complete original
    inventories, registered source hashes, qualification and prior-study lineage.
    Its numerical audit/replay entry point is never invoked by this renderer.
    """
    study, audit, audit_process, evaluation, reference, bla = (
        Path(p).resolve() for p in (study, audit, audit_process, evaluation, reference, bla))
    pins = {}

    def bind(key, expected):
        require(isinstance(expected, dict) and set(expected) == {'path', 'bytes', 'sha256'}, 'file descriptor required')
        require(descriptor(expected['path']) == expected, 'changed admitted file: '+key)
        pins[key] = expected
        return Path(expected['path'])

    bind('audit', descriptor(audit))
    bind('audit_process', descriptor(audit_process))
    closure = read_json(audit_process)
    require(closure.get('state') == 'EXITED' and closure.get('observed_exit_code') == 0
            and closure.get('success') is True and closure.get('error') is None
            and closure.get('closure_error') is None and closure.get('sources_unchanged') is True
            and closure.get('inputs_unchanged') is True and closure.get('audit_output') == pins['audit'],
            'successful original audit closure/output join required')
    require(finite(closure.get('elapsed_seconds'), positive=True)
            and closure['elapsed_seconds'] < closure.get('timeout_seconds', 0) == 3600,
            'original audit elapsed/cap')
    require(closure.get('sources_before') == closure.get('sources_after')
            and set(closure.get('sources_before', {})) == set(AUDITOR_SHA), 'held auditor/test source roster')
    for name, digest in AUDITOR_SHA.items():
        expected = closure['sources_before'][name]
        require(expected == descriptor(ROOT/name) and expected['sha256'] == digest, 'held audit source changed')
        bind('audit_source:'+name, expected)
    require(closure.get('inputs_before') == closure.get('inputs_after')
            and isinstance(closure.get('inputs_before'), dict), 'audit original input closure')
    for key, expected in closure['inputs_before'].items():
        bind('audit_input:'+key, expected)
    for key in ('helper', 'log', 'qualification', 'preflight', 'registration', 'freeze', 'fit_process', 'evaluation_process'):
        bind(key, closure.get(key))
    freeze_path = bind('auditor_freeze', descriptor(ROOT/AUDITOR_FREEZE))
    freeze = read_json(freeze_path)
    require(freeze.get('status') == 'FROZEN_BEFORE_EMPIRICAL_AUDIT'
            and freeze.get('source_sha256') == AUDITOR_SHA
            and freeze.get('qualification_sha256') == closure['qualification']['sha256']
            and freeze.get('qualification_runtime_preflight_sha256') == closure['preflight']['sha256']
            and freeze.get('registration_sha256') == closure['registration']['sha256'], 'published auditor freeze join')
    qualified, preflight = read_json(pins['qualification']['path']), read_json(pins['preflight']['path'])
    require(qualified.get('status') == 'PASS' and qualified.get('sources_unchanged') is True
            and qualified.get('sources_before') == qualified.get('sources_after') == closure['sources_before']
            and preflight.get('sources') == closure['sources_before']
            and qualified.get('preflight') == pins['preflight'], 'audit qualification/source join')
    for key, expected in preflight['snapshots'].items():
        require(expected['sha256'] == AUDITOR_SHA[key], 'qualified source snapshot mismatch')
        bind('audit_snapshot:'+key, expected)
    require([r['command'] for r in qualified['commands']] == preflight['commands']
            and len(qualified['commands']) == 2 and all(r['returncode'] == 0 for r in qualified['commands']),
            'original audit qualification commands')
    for i, row in enumerate(qualified['commands']):
        bind(f'audit_qualification_log:{i}', row['log'])
    result = read_json(audit)
    require(result.get('status') == 'PASS' and result.get('agreement') is True and result.get('study') == str(study)
            and closure.get('audit_status') == 'PASS' and closure.get('agreement') is True
            and closure.get('scientific_status') == result.get('scientific_status'), 'independent audit agreement required')
    fit_process, eval_process = Path(pins['fit_process']['path']), Path(pins['evaluation_process']['path'])
    command = closure.get('command')
    require(isinstance(command, list) and len(command) == 11 and command[1] == '-u'
            and command[3::2] == ['--study', '--process', '--evaluation-process', '--output']
            and (ROOT/command[0]).absolute() == (ROOT/'.venv/bin/python').absolute()
            and (ROOT/command[2]).resolve() == (ROOT/AUDITOR).resolve()
            and [(ROOT/v).resolve() for v in command[4::2]] == [study, fit_process, eval_process, audit],
            'original audit command identity')
    plan, admitted, prerequisites, fit_terminal, eval_terminal = admission.authenticate(study, fit_process, eval_process)
    require(admitted == result.get('inputs'), 'audit admission changed since numerical audit')
    require(fit_terminal['status'] == eval_terminal['status'] == 'completed'
            and fit_terminal['observed_exit_code'] == eval_terminal['observed_exit_code'] == 0,
            'closed original fit and evaluation required')
    require(admitted['evaluation_study'] == str(evaluation)
            and admitted['reference_summary']['path'] == str(reference/'summary.json')
            and prerequisites['bla_summary'] == bla/'summary.json', 'comparison directory identity')
    require(admitted['source'] == pins['audit_source:'+AUDITOR]
            and admitted['process'] == pins['fit_process'] and admitted['evaluation_process'] == pins['evaluation_process']
            and admitted['registration'] == pins['registration'], 'original process/source admission joins')
    for name, digest in plan['source_sha256'].items():
        value = descriptor(ROOT/name)
        require(value['sha256'] == digest, 'registered source changed')
        bind('registered_source:'+name, value)
    # Preserve the entire opaque audit input closure for checks after rendering.
    def descriptors(value, prefix):
        if isinstance(value, dict):
            if set(value) == {'path', 'bytes', 'sha256'}:
                bind(prefix, value)
            else:
                for key, child in value.items():
                    descriptors(child, prefix+'/'+key)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                descriptors(child, prefix+'/'+str(i))
    descriptors(admitted, 'admitted')
    for folder, files, prefix in ((study, admitted['files'], 'fit'), (evaluation, admitted['evaluation_files'], 'evaluation')):
        for name, value in files.items():
            bind(prefix+':'+name, {'path': str(folder/admission.relative(name)), **value})
    old_audit = read_json(prerequisites['bla_audit'])
    paths = {'nllfr_summary': evaluation/'summary.json', 'nllfr_evaluation': evaluation/'evaluation.json',
             'bla_summary': bla/'summary.json', 'bla_evaluation': bla/'evaluation.json',
             'reference_summary': reference/'summary.json', 'reference_evaluations': reference/'evaluations.json'}
    for key, path in paths.items():
        if key.startswith('nllfr_'):
            expected = {'path': str(path), **admitted['evaluation_files'][path.name]}
        elif key.startswith('bla_'):
            expected = {'path': str(path), **old_audit['inputs']['files'][path.name]}
        else:
            expected = admitted[key]
        bind(key, expected)
    return paths, pins


def check_audited_scalars(values, audit):
    summary, evaluation, saved = values['saved_nllfr_summary'], values['saved_nllfr_evaluation'], audit['results']
    require(summary['status'] == audit['scientific_status'] == saved['reference_status']
            and summary['fit_status'] == saved['fit_status'], 'audited scientific/fit status')
    for key, value in (('reference_mean_rmse', summary['mean_rmse']),
                       ('median_request_ms', summary['median_request_ms'])):
        require(saved[key] is None if value is None else finite(saved[key])
                and math.isclose(saved[key], value, rel_tol=1e-10, abs_tol=1e-12), 'audited scalar mismatch')
    indexed = {r['record_id']: r for r in saved['rows']}
    require(len(indexed) == 12, 'audited record count')
    for row in evaluation['rows']:
        other = indexed[row['record_id']]
        require(row['status'] == other['status'], 'audited record status')
        if row['status'] == 'complete':
            require(math.isclose(row['rmse'], other['rmse'], rel_tol=1e-10, abs_tol=1e-12), 'audited record RMSE')
    values['saved_audit_result'] = saved


def run(study, audit, audit_process, evaluation, reference, bla, output):
    study, audit, audit_process, evaluation, reference, bla, output = (
        Path(p).resolve() for p in (study, audit, audit_process, evaluation, reference, bla, output))
    require(not output.exists(), 'exclusive output required')
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output)
                for p in (study, evaluation, reference, bla, audit.parent, audit_process.parent)), 'output overlaps evidence')
    paths, pins = authenticate(study, audit, audit_process, evaluation, reference, bla)
    sources = {str(path): descriptor(path) for path in (Path(__file__).resolve(), Path(bla_plot.__file__).resolve(),
                                                        Path(bla_plot.reference_plot.__file__).resolve())}
    values = extract(*(read_json(paths[k]) for k in ('nllfr_summary', 'nllfr_evaluation', 'bla_summary',
                                                    'bla_evaluation', 'reference_summary', 'reference_evaluations')))
    check_audited_scalars(values, read_json(audit))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plotted-values.json', values)
    figure(values, output)
    require(all(descriptor(v['path']) == v for v in (*pins.values(), *sources.values())), 'evidence/source changed during render')
    outputs = {name: descriptor(output/name) for name in ('benchmark.png', 'benchmark.pdf', 'plotted-values.json')}
    write_json(output/'receipt.json', {'study': str(study), 'evaluation_study': str(evaluation),
        'reference_study': str(reference), 'bla_study': str(bla), 'inputs': pins, 'sources': sources,
        'outputs': outputs, 'scientific_status': values['scientific_status'], 'scope': SCOPE, 'timing_scope': TIMING_SCOPE})
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'audit', 'audit-process', 'evaluation', 'reference-study', 'bla-study', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    print(run(args.study, args.audit, args.audit_process, args.evaluation, args.reference_study, args.bla_study, args.output))
