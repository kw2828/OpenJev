"""Plot complete, closed factorial evidence using saved scalars only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import audit_fsm_author_factorial as admission

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = 'scripts/audit_fsm_author_factorial.py'
HELD = {
    AUDITOR: '450aeb0112d8931bd92527ae1d40eb55161b3fe2f0b65a20dda628fa49da248b',
    'tests/test_audit_fsm_author_factorial.py': '15923277d7853df840f65ee08d95f94f6c01a4c41583472aefeafbe829f0087a',
    'scripts/fsm_nllfr_budget_audit_math.py': 'd4454df6f8854d1895d7edfee75dd1e36a58ec616fb12537a7424359a538f8c6',
    'scripts/audit_fsm_author_nllfr.py': '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32',
}
CELLS = ('old16', 'old64', 'new16', 'new64')
COUNTS = {'forecast_attempts': 1536, 'record_slots': 48, 'warmups': 4, 'timed_requests': 96}
SCOPE = ('Exposed 100/200 mV DEV; C100/H128; equal mean of 12 correlated-period record RMSEs. '
         'Known future inputs, no future outputs in inference. Old 10,000-iteration fits remain diagnostic. '
         'The unchanged candidate is an error reference only; no candidate speed, control or novelty claim.')
TIMING_SCOPE = ('Current-host median of 24 rotated calls per cell, after one warmup each. '
                'Fresh legal copies, normalization, seed, full GN/SVD and physical forecast included; '
                'disk/scoring excluded. Same cache adapter for all four cells.')
require, read, close = admission.require, admission.read, admission.close


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular evidence file required')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def finite(value, *, positive=False):
    return type(value) in (float, int) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def authenticate(study, audit, audit_process):
    """Return ({audit, summary} paths, flat pins), without any array decoding.

    Reuse only the held auditor's metadata admission. Its numerical audit entry
    point is never called. Opaque parent/file hashing remains part of admission.
    """
    study, audit, audit_process = (Path(p).resolve() for p in (study, audit, audit_process))
    pins = {}

    def bind(key, expected):
        require(isinstance(expected, dict) and set(expected) == {'path', 'bytes', 'sha256'}
                and descriptor(expected['path']) == expected, 'changed evidence: '+key)
        pins[key] = expected
        return Path(expected['path'])

    bind('audit', descriptor(audit)); bind('audit_process', descriptor(audit_process))
    wrapper = read(audit_process)
    require(wrapper.get('state') == 'EXITED' and wrapper.get('observed_exit_code') == 0
            and wrapper.get('success') is True and wrapper.get('audit_status') == 'PASS'
            and wrapper.get('agreement') is True and wrapper.get('error') is None
            and wrapper.get('closure_error') is None and wrapper.get('evidence_errors') == {}
            and wrapper.get('sources_unchanged') is True and wrapper.get('inputs_unchanged') is True
            and wrapper.get('audit_output') == pins['audit'], 'closed successful original audit required')
    require(wrapper.get('sources_before') == wrapper.get('sources_after')
            and set(wrapper.get('sources_before', {})) == set(HELD), 'held audit source roster')
    for name, digest in HELD.items():
        value = wrapper['sources_before'][name]
        require(value == descriptor(ROOT/name) and value['sha256'] == digest, 'held audit source identity')
        bind('source:'+name, value)
    require(wrapper.get('inputs_before') == wrapper.get('inputs_after')
            and isinstance(wrapper.get('inputs_before'), dict), 'original audit input closure')
    for key, value in wrapper['inputs_before'].items():
        bind('audit_input:'+key, value)
    for key in ('helper', 'log', 'qualification', 'registration', 'freeze', 'evaluation_process'):
        bind(key, wrapper.get(key))
    process, freeze = Path(pins['evaluation_process']['path']), Path(pins['freeze']['path'])
    command = wrapper.get('command')
    require(isinstance(command, list) and len(command) == 11 and command[1] == '-u'
            and command[3::2] == ['--study', '--process', '--freeze', '--output']
            and (ROOT/command[0]).absolute() == (ROOT/'.venv/bin/python').absolute()
            and (ROOT/command[2]).resolve() == (ROOT/AUDITOR).resolve()
            and [(ROOT/p).resolve() for p in command[4::2]] == [study, process, freeze, audit],
            'original audit command identity')
    result = read(audit)
    require(result.get('status') == 'PASS' and result.get('agreement') is True
            and result.get('study') == str(study)
            and wrapper.get('scientific_status') == result.get('scientific_status'), 'audit agreement identity')
    plan, inputs, _paths, terminal, _parent, _prior = admission.authenticate(study, process, freeze)
    require(inputs == result.get('inputs') == wrapper.get('admission_before') == wrapper.get('admission_after')
            and terminal['status'] == 'completed' and terminal['observed_exit_code'] == 0,
            'closed original evaluation and audit admission required')
    for key in ('registration', 'freeze', 'qualification'):
        require(inputs[key] == pins[key], 'admitted '+key+' identity')
    require(inputs['process'] == pins['evaluation_process'], 'admitted evaluation process identity')
    for name, digest in plan['source_sha256'].items():
        value = descriptor(ROOT/name)
        require(value['sha256'] == digest, 'registered source drift')
        bind('registered_source:'+name, value)
    summary = study/'summary.json'
    bind('summary', {'path': str(summary), **inputs['files']['summary.json']})
    return {'audit': audit, 'summary': summary}, pins


def extract(audit, summary):
    """Validate the complete scalar matrix without deriving new forecast scores."""
    require(audit.get('status') == 'PASS' and audit.get('agreement') is True, 'audit PASS/agreement required')
    saved = audit['results']
    require(saved.get('matrix_status') == summary.get('status') == 'FACTORIAL_COMPLETE',
            'complete four-cell result required')
    require(set(saved['cells']) == set(summary['cells']) == set(CELLS) and summary['counts'] == COUNTS,
            'exact four-cell/count roster')
    require(summary['new_fit_status'] == saved['new_fit_status']
            and summary['new_fit_status'] in ('FIT_ONLY_COMPLETE', 'FIT_ONLY_INCOMPLETE')
            and type(summary['new_fit_iterations']) is int and summary['new_fit_iterations'] > 0
            and summary['new_fit_iterations'] == saved['new_fit_iterations'], 'actual training status/iterations')
    close(summary['continuation'], saved['continuation']); close(summary['contrasts'], saved['contrasts'])
    require(audit['scientific_status'] == ('REFERENCE_COMPLETE' if saved['continuation']['reference_complete']
            else 'REFERENCE_INCOMPLETE'), 'scientific completion identity')
    rows = []
    for cell in CELLS:
        row, compact = saved['cells'][cell], summary['cells'][cell]
        require(set(compact) == {'mean_rmse', 'amplitude_mean_rmse', 'median_request_ms',
                'persistent_numeric_bytes', 'context_status_counts', 'failure_counts'}, 'cell summary schema')
        close(compact, {k: row[k] for k in compact})
        require([r['record_id'] for r in row['rows']] == list(admission.DEV)
                and all(r['status'] == 'complete' and finite(r['rmse']) for r in row['rows']), 'twelve complete records')
        require(finite(row['mean_rmse']) and math.isclose(row['mean_rmse'], fmean(r['rmse'] for r in row['rows']),
                rel_tol=1e-10, abs_tol=1e-12) and finite(row['median_request_ms'], positive=True), 'finite complete metrics')
        require(row['failure_counts'] == dict.fromkeys(('forecast', 'record', 'warmup', 'timing'), 0)
                and sum(row['context_status_counts'].values()) == 384, 'complete forecast/cost coverage')
        old, directions = cell.startswith('old'), int(cell[-2:])
        training = 'Old 10,000-iteration fit (diagnostic)' if old else f'New {summary["new_fit_iterations"]:,}-iteration fit'
        if not old and summary['new_fit_status'] != 'FIT_ONLY_COMPLETE':
            training += ' (diagnostic)'
        rows.append({'cell': cell, 'label': training+'\n'+str(directions)+' context directions', **compact})
    candidate = saved['continuation']['candidate']
    candidate_error = saved['continuation']['family_means'][candidate]
    require(finite(candidate_error), 'unchanged candidate reference score')
    return {'cells': rows, 'candidate': candidate, 'candidate_rmse': candidate_error,
            'scientific_status': audit['scientific_status'], 'new_fit_status': summary['new_fit_status'],
            'new_fit_iterations': summary['new_fit_iterations'], 'scope': SCOPE, 'timing_scope': TIMING_SCOPE,
            'saved_summary': summary, 'saved_audit_counts': audit['counts']}


def figure(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8), sharey=True)
    colors = ('#9a6670', '#9a6670', '#28668b', '#28668b')
    for ax, field, title, xlabel in (
        (axes[0], 'mean_rmse', 'Exposed DEV forecast error', 'Equal-record standardized RMSE'),
        (axes[1], 'median_request_ms', 'Current-host full-request latency', 'Median latency, ms (24 calls per cell)'),
    ):
        numbers = [r[field] for r in values['cells']]
        if field == 'mean_rmse':
            numbers.append(values['candidate_rmse'])
            ax.axvline(values['candidate_rmse'], color='#6b7280', linestyle='--', linewidth=1.4,
                       label=f'Unchanged candidate error: {values["candidate_rmse"]:.5g}')
            ax.legend(loc='lower right', fontsize=8, frameon=False)
        for index, (row, color) in enumerate(zip(values['cells'], colors, strict=True)):
            ax.scatter(row[field], index, color=color, s=70, marker='D', zorder=3, clip_on=False)
            ax.annotate(f'{row[field]:.5g}', (row[field], index), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=10)
        ax.set_xlim(0, max(numbers)*1.28 if max(numbers) else 1.)
        ax.set_ylim(3.65, -.6)
        ax.set_title(title, pad=14); ax.set_xlabel(xlabel+' (lower is better)')
        ax.grid(axis='x', color='#e2e8f0'); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(4), [r['label'] for r in values['cells']])
    axes[0].tick_params(axis='y', length=0, pad=10)
    fig.suptitle('NL-LFR training and state-estimation budgets', fontsize=16, y=.97)
    fig.text(.025, .135, 'C100/H128; 12 exposed 100/200 mV DEV records. Same saved targets and standardized-difference scoring.', fontsize=9)
    fig.text(.025, .10, 'Known future inputs; no future outputs in inference. Old 10,000-iteration fit remains diagnostic.', fontsize=9)
    fig.text(.025, .065, 'Four cells share one timed cache adapter. Candidate error is a reference only; no candidate speed comparison.', fontsize=9)
    fig.text(.025, .03, 'Finite capped or stalled context solves remain scoreable. No untouched-test, control or novelty claim.', fontsize=9)
    fig.subplots_adjust(left=.29, right=.975, top=.84, bottom=.25, wspace=.28)
    for extension in ('png', 'pdf'):
        fig.savefig(Path(output)/f'benchmark.{extension}', dpi=180)
    plt.close(fig)


def run(study, audit, audit_process, output):
    study, audit, audit_process, output = (Path(p).resolve() for p in (study, audit, audit_process, output))
    require(not output.exists(), 'exclusive output required')
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output)
                for p in (study, audit.parent, audit_process.parent)), 'output overlaps evidence')
    paths, pins = authenticate(study, audit, audit_process)
    renderer = descriptor(__file__)
    values = extract(read(paths['audit']), read(paths['summary']))
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plotted-values.json', values)
    figure(values, output)
    require(all(descriptor(p['path']) == p for p in (*pins.values(), renderer)), 'evidence/source changed during render')
    outputs = {name: descriptor(output/name) for name in ('benchmark.png', 'benchmark.pdf', 'plotted-values.json')}
    write(output/'receipt.json', {'study': str(study), 'inputs': pins, 'renderer': renderer, 'outputs': outputs,
          'scientific_status': values['scientific_status'], 'scope': SCOPE, 'timing_scope': TIMING_SCOPE})
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'audit-process', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.audit, args.audit_process, args.output), indent=2))
