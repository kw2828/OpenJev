"""Presentation-only residual study plots from saved scalar JSON.

The caller must admit the original closed run and independent audit before
invocation. This helper does not load arrays, import models, execute forecasts,
fit parameters, select a learning rate, or recompute a scientific condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

ARCHITECTURES = ('affine_output_only', 'affine_feedback', 'tanh_output_only', 'tanh_feedback')
RATES = ('0.0001', '0.0003', '0.001')
SEEDS = (9201, 9202, 9203)
RECIPES = tuple(f'{arch}-lr{rate}' for arch in ARCHITECTURES for rate in RATES)
FAMILIES = (*RECIPES, 'native_varx')
RECORDS = {f'{amp}-realization-{realization}-period-{period}'
           for amp in ('100mV', '200mV') for realization in (3, 4, 5) for period in (0, 1)}
CONDITIONS = ('all_36_fits_complete', 'all_37_evaluations_complete',
              'initial_identity_and_frozen_backbone', 'five_percent_below_strongest_control',
              'every_seed_below_three_selected_controls', 'no_record_over_two_percent_strongest_control',
              'both_amplitudes_below_strongest_control', 'latency_within_ten_percent_tanh_output',
              'storage_no_more_than_tanh_output')
NAMES = {'affine_output_only': 'Affine output only', 'affine_feedback': 'Affine feedback',
         'tanh_output_only': 'Tanh output only', 'tanh_feedback': 'Tanh feedback'}
COLORS = {'affine_output_only': '#367393', 'affine_feedback': '#5c5d99',
          'tanh_output_only': '#4c8b68', 'tanh_feedback': '#ba6927', 'native_varx': '#676b70'}


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


def index_records(rows):
    require(isinstance(rows, list), 'saved evaluation rows required')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and row.get('record_id') in RECORDS, 'unknown record identity')
        name = row['record_id']
        require(name not in indexed, 'duplicate record identity')
        require(row.get('status') in ('complete', 'failed', 'not_run'), 'unknown record status')
        if row['status'] == 'complete':
            require(nonnegative(row.get('rmse')) and row.get('requests') == 32
                    and row.get('horizon') == 128, 'invalid complete score or request geometry')
        else:
            require(isinstance(row.get('error'), str) and bool(row['error']), 'failed row needs reason')
        indexed[name] = row
    return indexed


def extract(summary, evaluations, fits):
    """Validate saved scalar joins and retain all 37 instance slots, without selection."""
    require(isinstance(summary, dict) and summary.get('status') in ('DEVELOPMENT_PASS', 'DEVELOPMENT_FAIL'),
            'saved development outcome required')
    require(set(summary.get('families', {})) == set(FAMILIES), 'exact thirteen-family summary required')
    conditions = summary.get('conditions')
    require(isinstance(conditions, dict) and set(conditions) == set(CONDITIONS)
            and all(type(value) is bool for value in conditions.values()), 'nine saved conditions required')
    require(summary.get('total') == 9 and summary.get('passed') == sum(conditions.values()),
            'saved condition count mismatch')
    require((summary['status'] == 'DEVELOPMENT_PASS') == all(conditions.values()), 'saved outcome mismatch')
    selected = summary.get('selected_by_architecture')
    require(isinstance(selected, dict) and set(selected) == set(ARCHITECTURES), 'four saved selected-rate slots required')
    for arch, family in selected.items():
        require(family is None or family in {f'{arch}-lr{rate}' for rate in RATES}, 'invalid saved selected recipe')
    require(isinstance(fits, list) and isinstance(evaluations, list), 'fit/evaluation lists required')
    fit_index, evaluation_index = {}, {}
    for fit in fits:
        key = fit.get('family'), fit.get('seed')
        require(key[0] in RECIPES and key[1] in SEEDS and key not in fit_index, 'invalid or duplicated fit identity')
        require(fit.get('status') in ('complete', 'failed'), 'unknown fit status')
        require(type(fit.get('accepted_updates')) is int and 0 <= fit['accepted_updates'] <= 2048,
                'invalid saved update count')
        if fit['status'] == 'complete':
            require(fit['accepted_updates'] == 2048 and fit.get('error') is None, 'incomplete successful fit')
        else:
            require(isinstance(fit.get('error'), dict) and bool(fit['error']), 'failed fit needs reason')
        fit_index[key] = fit
    for evaluation in evaluations:
        key = evaluation.get('family'), evaluation.get('seed')
        require(key[0] in FAMILIES and (key[1] in SEEDS if key[0] in RECIPES else key[1] is None)
                and key not in evaluation_index, 'invalid or duplicated evaluation identity')
        index_records(evaluation.get('rows'))
        evaluation_index[key] = evaluation
    families = []
    for family in FAMILIES:
        native = family == 'native_varx'
        arch, rate = ('native_varx', None) if native else family.rsplit('-lr', 1)
        saved = summary['families'][family]
        require(all(type(saved.get(key)) is bool for key in
                    ('eligible', 'score_eligible', 'latency_eligible', 'storage_eligible')), 'saved eligibility flags required')
        members = []
        for seed in ((None,) if native else SEEDS):
            key = family, seed
            fit, evaluation = fit_index.get(key), evaluation_index.get(key)
            member = {'seed': seed, 'fit_status': 'reference' if native else fit['status'] if fit else 'missing',
                      'accepted_updates': None if fit is None else fit['accepted_updates'],
                      'fit_error': None if fit is None else fit.get('error'),
                      'score_status': 'missing', 'rmse': None, 'complete_records': 0,
                      'latency_ms': None, 'timing_error': 'missing evaluation', 'records': []}
            if evaluation is not None:
                records = index_records(evaluation['rows'])
                complete = [row for row in records.values() if row['status'] == 'complete']
                member.update(records=evaluation['rows'], complete_records=len(complete),
                              score_status='complete' if len(complete) == 12 else 'incomplete',
                              timing_error=evaluation.get('timing_error'))
                if len(complete) == 12:
                    member['rmse'] = fmean(row['rmse'] for row in complete)
                if evaluation.get('timing_error') is None:
                    samples = evaluation.get('request_ms')
                    require(isinstance(samples, list) and len(samples) == 24 and all(positive(t) for t in samples)
                            and positive(evaluation.get('median_request_ms')), 'successful timing requires 24 positive samples')
                    member['latency_ms'] = evaluation['median_request_ms']
                else:
                    require(isinstance(evaluation['timing_error'], str) and bool(evaluation['timing_error']),
                            'timing failure needs reason')
            members.append(member)
        for flag, key, metric in (('score_eligible', 'mean_rmse', 'rmse'),
                                  ('latency_eligible', 'mean_seed_median_ms', 'latency_ms')):
            if saved[flag]:
                require(all(m[metric] is not None for m in members), 'eligible summary has missing instance values')
                if flag == 'score_eligible':
                    require(native or all(m['fit_status'] == 'complete' for m in members), 'eligible score has failed/missing fit')
                require(nonnegative(saved.get(key)) and math.isclose(saved[key], fmean(m[metric] for m in members),
                                                                    rel_tol=1e-10, abs_tol=1e-12), 'saved scalar join mismatch: ' + family + '/' + key)
        label = 'Frozen VARX 32' if native else f'{NAMES[arch]}   |   LR {rate}'
        families.append({'family': family, 'architecture': arch, 'rate': rate, 'label': label,
                         'selected': not native and selected[arch] == family,
                         'eligible': saved['eligible'], 'members': members,
                         'mean_rmse': saved.get('mean_rmse') if saved['score_eligible'] else None,
                         'mean_seed_median_ms': saved.get('mean_seed_median_ms') if saved['latency_eligible'] else None})
    return {'status': summary['status'], 'passed': summary['passed'], 'total': summary['total'],
            'conditions': conditions, 'selected_by_architecture': selected,
            'candidate': summary.get('candidate'), 'strongest_affine': summary.get('strongest_affine'),
            'strongest_control': summary.get('strongest_control'), 'families': families,
            'saved_fits': fits, 'saved_evaluations': evaluations, 'saved_summary': summary,
            'scope': 'Development on previously exposed FSM estimation data; C100/H128, 12 records with correlated periods. No confirmation or official benchmark claim.',
            'point_semantics': 'Three seed means and a pooled equal-record mean; markers are not confidence intervals. Latency diamonds average seed medians.',
            'timing_scope': 'One warm-up followed by 24 timed full requests per seed; each includes normalization, float64 conversion, condition, 128-step forecast and physical output scaling; excludes loading and disk I/O.',
            'dtype': 'Float64 for all residual and VARX models.', 'no_model_or_rule_replay': True}


def axis_spec(numbers):
    require(all(nonnegative(value) for value in numbers), 'finite nonnegative plot values required')
    positive_values = [value for value in numbers if value > 0]
    if not positive_values:
        return {'scale': 'linear', 'limits': (0., 1.), 'linthresh': None}
    low, high = min(positive_values), max(positive_values)
    if high / low > 100:
        return {'scale': 'symlog', 'limits': (0., high * 1.25), 'linthresh': low * .5}
    return {'scale': 'linear', 'limits': (0., high * 1.15), 'linthresh': None}


def render(values, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.4), sharey=True)
    markers = {9201: 'o', 9202: 's', 9203: '^', None: 'o'}
    for ax, metric, mean_key, title in zip(axes, ('rmse', 'latency_ms'),
            ('mean_rmse', 'mean_seed_median_ms'), ('Forecast error', 'Full request latency'), strict=True):
        numbers = [m[metric] for f in values['families'] for m in f['members'] if m[metric] is not None]
        numbers += [f[mean_key] for f in values['families'] if f[mean_key] is not None]
        scale = axis_spec(numbers)
        ax.set_xscale(scale['scale'], **({'linthresh': scale['linthresh']} if scale['scale'] == 'symlog' else {}))
        ax.set_xlim(*scale['limits'])
        for i, family in enumerate(values['families']):
            color = COLORS[family['architecture']]
            if family['selected']:
                ax.axhspan(i-.44, i+.44, color=color, alpha=.09, zorder=0)
            members = family['members']
            offsets = (-.18, 0., .18) if len(members) == 3 else (0.,)
            missing = []
            for member, offset in zip(members, offsets, strict=True):
                value = member[metric]
                if value is None:
                    reason = member['fit_status'] if member['fit_status'] in ('failed', 'missing') else 'unavailable'
                    missing.append(f"{member['seed'] or 'reference'} ({reason})")
                else:
                    ax.scatter(value, i+offset, color=color, marker=markers[member['seed']], s=37,
                               edgecolors='white', linewidths=.45, zorder=4)
            if family[mean_key] is not None:
                ax.scatter(family[mean_key], i, marker='D', s=66, facecolors='none', edgecolors='black', linewidths=1.2, zorder=5)
            elif any(member[metric] is not None for member in members):
                ax.text(.99, i+.34, 'No complete pooled value', transform=ax.get_yaxis_transform(),
                        ha='right', va='center', color='#8a3f29', fontsize=7.5)
            if missing:
                ax.text(.99, i-.27, '; '.join(missing), transform=ax.get_yaxis_transform(),
                        ha='right', va='center', color='#8a3f29', fontsize=7.1)
        labels = [f['label'] + ('  [selected]' if f['selected'] else '') for f in values['families']]
        ax.set_yticks(range(len(FAMILIES)), labels)
        if ax is axes[0]:
            for label, family in zip(ax.get_yticklabels(), values['families'], strict=True):
                label.set_fontweight('bold' if family['selected'] else 'normal')
        for boundary in (2.5, 5.5, 8.5, 11.5):
            ax.axhline(boundary, color='#c6c8ca', linewidth=.7)
        ax.set_ylim(len(FAMILIES)-.45, -.65)
        ax.grid(axis='x', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc='left', pad=12, fontweight='bold')
        units = 'Mean record RMSE, FIT-standardized units' if metric == 'rmse' else 'Request latency, ms'
        scale_label = '; symlog scale' if scale['scale'] == 'symlog' else ''
        ax.set_xlabel(units + '\nLower is better; zero included' + scale_label)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.suptitle(f"FSM residual study: {values['status']} ({values['passed']}/{values['total']} checks)",
                 x=.045, ha='left', fontsize=18, fontweight='bold')
    fig.text(.045, .933, 'All 12 residual recipes and the frozen VARX baseline. Shaded rows mark each architecture\'s saved DEV-selected rate.', fontsize=11)
    legend = [Line2D([0], [0], marker=markers[seed], linestyle='none', color='#367393', label=f'Seed {seed}') for seed in SEEDS]
    legend.append(Line2D([0], [0], marker='D', linestyle='none', markerfacecolor='none', color='black', label='Complete pooled mean'))
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.61, .088), ncol=4, frameon=False, fontsize=9)
    fig.text(.045, .065, '100 observations / 128-step forecast. Twelve development records include correlated periods; all models use float64.', fontsize=9)
    fig.text(.045, .045, 'Seed markers are not confidence intervals. Latency: one warm-up, then 24 timed requests per seed; diamonds average seed medians. Missing or failed slots stay labeled.', fontsize=9)
    fig.text(.045, .025, 'DEV selects one rate per architecture. All rates remain visible. No new confirmation, official benchmark or novel-architecture claim.', fontsize=9)
    fig.subplots_adjust(left=.31, right=.98, top=.88, bottom=.18, wspace=.17)
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
    paths = tuple(args.study/name for name in ('summary.json', 'evaluations.json', 'fits.json'))
    inputs = {path.name: descriptor(path) for path in paths}
    renderer = descriptor(__file__)
    values = extract(*(read_json(path) for path in paths))
    args.output.mkdir(parents=True, exist_ok=False)
    figures = render(values, args.output)
    plotted = args.output/'plotted-values.json'
    write_json(plotted, values)
    require(inputs == {path.name: descriptor(path) for path in paths}, 'input scalar files changed during render')
    require(renderer == descriptor(__file__), 'renderer changed during render')
    write_json(args.output/'plot-receipt.json', {'study': str(args.study.resolve()), 'inputs': inputs,
               'renderer': renderer, 'outputs': {path.name: descriptor(path) for path in (*figures, plotted)},
               'scope': 'Presentation-only saved scalars after caller-admitted original run/audit. No model/array/scientific condition replay.'})


if __name__ == '__main__':
    main()
