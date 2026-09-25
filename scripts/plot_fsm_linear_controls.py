"""Present saved FSM linear-control scalars after caller-admitted run/audit closure.

No measurement or prediction arrays, model imports, fitting, inference, selection,
or scientific-rule reconstruction. All 15 families and 25 instance slots remain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

SEEDS = (9201, 9202, 9203)
LINEAR = tuple(f'varx{order}-ridge{alpha}' for order in (32, 64, 96)
               for alpha in ('1e-06', '0.001', '0.1'))
RESIDUALS = ('affine_output_only-lr0.0001', 'affine_feedback-lr0.0001',
             'tanh_output_only-lr0.001', 'tanh_feedback-lr0.0003')
FOLDED, NATIVE = 'folded_affine_feedback', 'native_varx'
FAMILIES = (*LINEAR, *RESIDUALS, FOLDED, NATIVE)
SEEDED = (*RESIDUALS, FOLDED)
CANDIDATE = RESIDUALS[-1]
RECORDS = {f'{amp}-realization-{realization}-period-{period}'
           for amp in ('100mV', '200mV') for realization in (3, 4, 5) for period in (0, 1)}
CONDITIONS = ('all_9_fits_complete', 'all_25_evaluations_complete', 'all_3_folds_equivalent',
              'five_percent_below_strongest_control', 'every_seed_below_strongest_control',
              'no_record_over_two_percent_strongest_control', 'both_amplitudes_below_strongest_control',
              'latency_within_ten_percent_tanh_output', 'storage_no_more_than_tanh_output')
LABELS = dict(zip(RESIDUALS, ('Affine output only | LR 1e-4', 'Affine feedback | LR 1e-4',
                            'Tanh output only | LR 1e-3', 'Tanh feedback | LR 3e-4'), strict=True))
LABELS.update({FOLDED: 'Folded affine feedback', NATIVE: 'Original native VARX 32'})
for _family in LINEAR:
    _order, _alpha = _family.removeprefix('varx').split('-ridge')
    LABELS[_family] = f'VARX {_order} | ridge {_alpha}'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def positive(value):
    return nonnegative(value) and value > 0


def descriptor(path):
    path = Path(path)
    content = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON token ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def records(rows):
    require(isinstance(rows, list), 'saved record rows required')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and row.get('record_id') in RECORDS, 'unknown record')
        require(row['record_id'] not in indexed, 'duplicate record')
        require(row.get('status') in ('complete', 'failed', 'not_run'), 'unknown record status')
        if row['status'] == 'complete':
            require(nonnegative(row.get('rmse')) and row.get('requests') == 32
                    and row.get('horizon') == 128, 'invalid score or request geometry')
        else:
            require(isinstance(row.get('error'), str) and bool(row['error']), 'failure reason required')
        indexed[row['record_id']] = row
    return indexed


def extract(summary, evaluations):
    """Join saved means with scalar records; retain missing and failed slots."""
    require(summary.get('status') in ('DEVELOPMENT_PASS', 'DEVELOPMENT_FAIL'), 'saved outcome required')
    require(set(summary.get('families', {})) == set(FAMILIES), 'exact fifteen-family summary required')
    conditions = summary.get('conditions')
    require(isinstance(conditions, dict) and set(conditions) == set(CONDITIONS)
            and all(type(v) is bool for v in conditions.values()), 'nine saved conditions required')
    require(summary.get('passed') == sum(conditions.values()) and summary.get('total') == 9
            and (summary['status'] == 'DEVELOPMENT_PASS') == all(conditions.values()), 'outcome/count mismatch')
    require(summary.get('candidate') == CANDIDATE, 'fixed candidate identity mismatch')
    require(summary.get('strongest_control') in (None, *(f for f in FAMILIES if f != CANDIDATE)),
            'invalid saved strongest control')
    require(isinstance(evaluations, list), 'saved evaluation list required')
    indexed = {}
    for evaluation in evaluations:
        key = evaluation.get('family'), evaluation.get('seed')
        require(key[0] in FAMILIES and key not in indexed, 'unknown/duplicate evaluation')
        require(key[1] in SEEDS if key[0] in SEEDED else key[1] is None, 'invalid seed')
        records(evaluation.get('rows'))
        indexed[key] = evaluation
    families = []
    for family in FAMILIES:
        saved = summary['families'][family]
        flags = ('eligible', 'score_eligible', 'latency_eligible', 'storage_eligible')
        require(all(type(saved.get(k)) is bool for k in flags), 'eligibility flags required')
        require(saved['eligible'] == all(saved[k] for k in flags[1:]), 'inconsistent eligibility')
        members = []
        for seed in (SEEDS if family in SEEDED else (None,)):
            evaluation = indexed.get((family, seed))
            member = {'seed': seed, 'score_status': 'missing', 'complete_records': 0,
                      'rmse': None, 'latency_ms': None, 'timing_error': 'missing evaluation',
                      'persistent_numeric_bytes': None, 'records': []}
            if evaluation is not None:
                rows = records(evaluation['rows'])
                complete = [r for r in rows.values() if r['status'] == 'complete']
                member.update(score_status='complete' if len(complete) == 12 else 'incomplete',
                              complete_records=len(complete), records=evaluation['rows'],
                              persistent_numeric_bytes=evaluation.get('persistent_numeric_bytes'),
                              timing_error=evaluation.get('timing_error'))
                if len(complete) == 12:
                    member['rmse'] = fmean(r['rmse'] for r in complete)
                samples = evaluation.get('request_ms')
                if member['timing_error'] is None:
                    if (isinstance(samples, list) and len(samples) == 24 and all(positive(t) for t in samples)
                            and positive(evaluation.get('median_request_ms'))):
                        member['latency_ms'] = evaluation['median_request_ms']
                    else:
                        member['timing_error'] = 'invalid saved timing values or sample count'
                else:
                    require(isinstance(member['timing_error'], str) and bool(member['timing_error']),
                            'timing failure reason required')
            members.append(member)
        for flag, saved_key, point_key in (('score_eligible', 'mean_rmse', 'rmse'),
                                          ('latency_eligible', 'mean_seed_median_ms', 'latency_ms')):
            if saved[flag]:
                require(all(m[point_key] is not None for m in members), 'eligible family has missing values')
                require(nonnegative(saved.get(saved_key)) and math.isclose(saved[saved_key],
                        fmean(m[point_key] for m in members), rel_tol=1e-10, abs_tol=1e-12),
                        'saved mean mismatch: ' + family + '/' + saved_key)
        if saved['storage_eligible']:
            require(all(type(m['persistent_numeric_bytes']) is int and m['persistent_numeric_bytes'] > 0
                        for m in members), 'eligible family has invalid storage')
            require(saved.get('persistent_numeric_bytes') == max(m['persistent_numeric_bytes'] for m in members),
                    'saved storage mismatch')
        families.append({'family': family, 'label': LABELS[family], 'members': members,
                         'candidate': family == CANDIDATE, 'strongest_control': family == summary['strongest_control'],
                         'eligible': saved['eligible'], 'score_eligible': saved['score_eligible'],
                         'latency_eligible': saved['latency_eligible'],
                         'mean_rmse': saved.get('mean_rmse') if saved['score_eligible'] else None,
                         'mean_seed_median_ms': saved.get('mean_seed_median_ms') if saved['latency_eligible'] else None,
                         'persistent_numeric_bytes': saved.get('persistent_numeric_bytes') if saved['storage_eligible'] else None})
    return {'status': summary['status'], 'passed': summary['passed'], 'total': summary['total'],
            'conditions': conditions, 'families': families, 'saved_summary': summary,
            'saved_evaluations': evaluations, 'no_model_or_rule_replay': True,
            'scope': 'Exposed 100/200mV DEV, C100/H128, 12 correlated-period records. Candidate and all inherited checkpoints unchanged. No confirmation or official benchmark claim.',
            'point_semantics': 'Five families retain three seed points each; nine fresh VARX controls and one original VARX have one point each. Diamonds show complete family means; seed points are not confidence intervals.',
            'timing_scope': 'One warm-up and 24 full requests per instance. Means of saved instance medians; normalization, conversion, condition, rollout, validation and denormalization included. Loading and disk I/O excluded.',
            'timing_limit': 'Fixed serial order; small timing differences do not establish speed improvements. Native BLAS threading was not forced.'}


def axis_spec(numbers):
    require(all(nonnegative(v) for v in numbers), 'finite nonnegative values required')
    positives = [v for v in numbers if v > 0]
    if not positives:
        return {'scale': 'linear', 'limits': (0., 1.), 'linthresh': None}
    low, high = min(positives), max(positives)
    if high / low > 100:
        return {'scale': 'symlog', 'limits': (0., high*1.25), 'linthresh': low*.5}
    return {'scale': 'linear', 'limits': (0., high*1.15), 'linthresh': None}


def render(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(18, 11.5), sharey=True)
    markers = {9201: 'o', 9202: 's', 9203: '^', None: 'o'}
    for ax, metric, mean_key, title in zip(axes, ('rmse', 'latency_ms'),
            ('mean_rmse', 'mean_seed_median_ms'), ('Forecast error', 'Full request latency'), strict=True):
        numbers = [m[metric] for f in values['families'] for m in f['members'] if m[metric] is not None]
        numbers += [f[mean_key] for f in values['families'] if f[mean_key] is not None]
        spec = axis_spec(numbers)
        ax.set_xscale(spec['scale'], **({'linthresh': spec['linthresh']} if spec['scale'] == 'symlog' else {}))
        ax.set_xlim(*spec['limits'])
        for i, family in enumerate(values['families']):
            color = '#ba6927' if family['candidate'] else '#387f73' if family['strongest_control'] else '#4c7292'
            if family['candidate'] or family['strongest_control']:
                ax.axhspan(i-.44, i+.44, color=color, alpha=.10, zorder=0)
            offsets = (-.18, 0., .18) if len(family['members']) == 3 else (0.,)
            missing = []
            for member, offset in zip(family['members'], offsets, strict=True):
                value = member[metric]
                if value is None:
                    reason = member['score_status'] if metric == 'rmse' else 'cost unavailable'
                    missing.append(f"{member['seed'] or 'reference'}: {reason}")
                else:
                    ax.scatter(value, i+offset, color=color, marker=markers[member['seed']], s=40,
                               edgecolors='white', linewidths=.45, zorder=4, clip_on=False)
            if family[mean_key] is not None:
                ax.scatter(family[mean_key], i, marker='D', s=67, facecolors='none',
                           edgecolors='black', linewidths=1.2, zorder=5, clip_on=False)
            elif any(m[metric] is not None for m in family['members']):
                ax.text(.99, i+.32, 'No eligible pooled value', transform=ax.get_yaxis_transform(),
                        ha='right', color='#8a3f29', fontsize=7.5)
            if missing:
                ax.text(.99, i-.27, '; '.join(missing), transform=ax.get_yaxis_transform(),
                        ha='right', color='#8a3f29', fontsize=7.1)
        labels = [f['label'] + ('  [candidate]' if f['candidate'] else '  [strongest control]' if f['strongest_control'] else '')
                  + ('' if f['eligible'] else '  [ineligible]') for f in values['families']]
        ax.set_yticks(range(len(FAMILIES)), labels)
        for boundary in (2.5, 5.5, 8.5, 12.5, 13.5):
            ax.axhline(boundary, color='#c6c8ca', linewidth=.7)
        ax.set_ylim(len(FAMILIES)-.45, -.65)
        ax.grid(axis='x', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        units = 'Mean record RMSE, FIT-standardized units' if metric == 'rmse' else 'Request latency, ms'
        ax.set_xlabel(units + '\nLower is better; zero included' + ('; symlog scale' if spec['scale'] == 'symlog' else ''))
    axes[1].tick_params(axis='y', labelleft=False)
    fig.suptitle(f"FSM stronger linear controls: {values['status']} ({values['passed']}/{values['total']} checks)",
                 x=.035, ha='left', fontsize=18, fontweight='bold')
    fig.text(.035, .935, 'All 15 families and 25 instances. Fixed inherited candidate in orange; saved globally strongest reference in green.', fontsize=11)
    legend = [Line2D([0], [0], marker=markers[s], linestyle='none', color='#4c7292', label=f'Seed {s}') for s in SEEDS]
    legend.append(Line2D([0], [0], marker='D', linestyle='none', markerfacecolor='none', color='black', label='Complete family mean'))
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.65, .087), ncol=4, frameon=False, fontsize=9)
    fig.text(.035, .065, 'Exposed 100/200 mV DEV; 100 observations, 128-step forecast, 12 records with correlated periods. Seed markers are not confidence intervals.', fontsize=9)
    fig.text(.035, .045, 'Complete requests include normalization, conditioning, forecast and output scaling. Fixed serial timing; small differences are not architectural speed gains.', fontsize=9)
    fig.text(.035, .025, 'No new neural updates. All controls remain visible, including failed or missing slots. No confirmation, official benchmark or novelty claim.', fontsize=9)
    fig.subplots_adjust(left=.36, right=.98, top=.89, bottom=.18, wspace=.18)
    paths = (output/'benchmark.png', output/'benchmark.pdf')
    for path in paths:
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = tuple(args.study/name for name in ('summary.json', 'evaluations.json'))
    inputs = {path.name: descriptor(path) for path in paths}
    renderer = descriptor(__file__)
    values = extract(*(read_json(path) for path in paths))
    args.output.mkdir(parents=True, exist_ok=False)
    figures = render(values, args.output)
    plotted = args.output/'plotted-values.json'
    write_json(plotted, values)
    require(inputs == {p.name: descriptor(p) for p in paths}, 'input scalars changed during render')
    require(renderer == descriptor(__file__), 'renderer changed during render')
    write_json(args.output/'plot-receipt.json', {'study': str(args.study.resolve()), 'inputs': inputs,
               'renderer': renderer, 'outputs': {p.name: descriptor(p) for p in (*figures, plotted)},
               'scope': 'Saved scalar presentation after caller-admitted original run/audit. No arrays, model execution or rule replay.'})


if __name__ == '__main__':
    main()
