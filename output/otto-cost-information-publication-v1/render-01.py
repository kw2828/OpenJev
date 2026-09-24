"""Render a prospectively registered diagnostic only after original closure.

Reads authenticated JSON and opaque file hashes. No arrays, checkpoints, model,
teacher or simulation are decoded or executed. Plotting follows authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'output/otto-cost-information-v1'
DEST = ROOT / 'research/otto-cost-information-results'
DOCUMENT = ROOT / 'research/otto-cost-information-results.md'
FAMILIES = ('effect_recurrent', 'paired_recurrent', 'paired_blind', 'paired_direct')
LABELS = {'effect_recurrent': 'Effect recurrent', 'paired_recurrent': 'Paired recurrent',
          'paired_blind': 'Paired action blind', 'paired_direct': 'Paired direct'}
SEEDS = (330000001, 330000002, 330000003)
REGIMES = ('lambda3', 'lambda4')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), 'regular saved artifact')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def consume(directory, receipt, name):
    require(descriptor(directory / name) == receipt['files'][name], 'unchanged closed JSON ' + name)
    return read(directory / name)


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'finite saved publication value')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def number(value, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), 'finite saved number')
    return float(value)


def average(values):
    require(bool(values), 'nonempty descriptive average')
    return math.fsum(number(value) / len(values) for value in values)


def authenticate(closure_sha):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', closure_sha), 'root and external closure SHA256')
    closure_path, registration = STUDY / 'closure-01.json', STUDY / 'registration-01.json'
    require(descriptor(closure_path)['sha256'] == closure_sha, 'externally supplied original closure')
    closure = read(closure_path)
    require(closure['version'] == 'otto-cost-information-v1' and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True
            and closure['registration'] == descriptor(registration)
            and closure['old_test_admitted'] is False and closure['new_execution_admitted'] is False
            and closure['new_training'] is False and closure['architecture_claim'] is False
            and set(closure['processes']) == {'collection', 'prediction', 'audit'}, 'complete diagnostic-only closure')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import otto_cost_information_common as c

    plan = read(registration)
    c.check_plan(plan)
    receipts, previous = {}, None
    for phase, directory, prefix, expected_phase in (
            ('collection', 'collection-01', 'collection-native-01', 'collect'),
            ('prediction', 'prediction-01', 'prediction-native-01', 'predict'),
            ('audit', 'audit-01', 'audit-native-01', 'audit')):
        receipt_path, terminal_path = STUDY / directory / 'receipt.json', STUDY / (prefix + '.terminal.json')
        require(descriptor(receipt_path) == closure['processes'][phase]['receipt']
                and descriptor(terminal_path) == closure['processes'][phase]['terminal'], 'original phase byte joins')
        receipt, terminal = c.closed(STUDY / directory, terminal_path), read(terminal_path)
        require(receipt['plan_sha256'] == closure['registration']['sha256'] and receipt['phase'] == expected_phase
                and receipt['old_test_decodes'] == receipt['astra_calls'] == 0
                and receipt['wall_seconds'] == closure['processes'][phase]['worker_seconds'], 'registered original phase')
        require(previous is None or previous <= terminal['started_ns'], 'original phase ordering')
        previous = terminal['finished_ns']
        receipts[phase] = receipt
    audit = consume(STUDY / 'audit-01', receipts['audit'], 'audit.json')
    prediction = consume(STUDY / 'prediction-01', receipts['prediction'], 'summary.json')
    collection = consume(STUDY / 'collection-01', receipts['collection'], 'summary.json')
    require(descriptor(STUDY / 'audit-01/audit.json') == closure['audit']
            and audit['agreement'] is True and audit['technical_complete'] is False
            and audit['requires_original_supervisor_closure'] is True
            and audit['gate'] == closure['gate'] == audit['report']['gate']
            and prediction['status'] == closure['status']
            and audit['inputs']['collection_receipt'] == closure['processes']['collection']['receipt']
            and audit['inputs']['prediction_receipt'] == closure['processes']['prediction']['receipt'], 'independent audit and producer joins')
    require(prediction['parent_reference'] == collection['parent_reference'] == closure['parent_reference']
            and prediction['calls']['training_updates'] == 0 and collection['native_continuation_steps'] == 0,
            'same frozen parents and no new training or actual continuation')
    report, gate = audit['report'], audit['gate']
    require(report['new_training'] is False and report['original_action_effect_gate_unchanged'] == 'DEV_FAIL 6/18'
            and gate['admits_model_training'] is False and gate['architecture_claim'] is False
            and gate['status'] == closure['status'] == ('HEADROOM_RESOLVED' if gate['passed'] else 'HEADROOM_NOT_RESOLVED'),
            'unchanged failed parent and current diagnostic status')
    require(len(report['rows']) == 48 and len(report['decompositions']) == 24, 'all frozen policy/horizon views')
    keys = {(row['family'], row['fit_seed'], row['regime'], row['horizon']) for row in report['rows']}
    require(len(keys) == 48 and keys == {(f, s, r, h) for f in FAMILIES for s in SEEDS for r in REGIMES for h in (4, 8)}, 'complete descriptive roster')
    require(len(gate['groups']) == 2 and [row['regime'] for row in gate['groups']] == list(REGIMES)
            and gate['passed'] is all(row['passed'] for row in gate['groups']), 'complete primary settings')
    require(len({row['case_id'] for row in gate['cases']}) == len(gate['cases']) == collection['retained'], 'complete primary case support')
    for group in gate['groups']:
        require(group['cases'] == collection['counts'][group['regime']]
                and group['interval_columns'] == ['control_gap', 'gain', 'gain_minus_required']
                and len(group['approximate_95_percent_interval']) == 3
                and all(len(interval) == 2 and interval[0] <= interval[1] for interval in group['approximate_95_percent_interval'])
                and group['passed'] is all(group['conditions'].values()), 'primary interval and condition roster')
    for row in report['rows']:
        require(row['cases'] == collection['counts'][row['regime']], 'all retained cases in descriptive denominator')
        for key in ('total_regret', 'information_advantage', 'approximation_regret'):
            number(row[key], nonnegative=True)
    for value in (closure, audit, prediction, collection):
        finite_tree(value)
    return closure, report, prediction, collection, plan


def extract(closure, report, prediction, collection, plan):
    groups = [{key: row[key] for key in ('regime', 'cases', 'control_gap', 'reference_gap', 'gain',
               'required_fraction', 'gain_minus_required', 'approximate_95_percent_interval', 'conditions', 'passed')}
              for row in report['gate']['groups']]
    support = []
    minimum = plan['config']['min_mc_survivors']
    for regime in REGIMES:
        cases = [row for row in report['gate']['cases'] if row['regime'] == regime]
        positive = [row for row in cases if row['exact_survival'] > 0]
        support.append({'regime': regime, 'retained_prefixes': len(cases), 'exact_zero_survival_prefixes': len(cases) - len(positive),
            'positive_survival_below_required_draws': sum(min(row['select_survivors'], row['eval_survivors']) < minimum for row in positive),
            'minimum_select_survivors': min(row['select_survivors'] for row in cases),
            'minimum_eval_survivors': min(row['eval_survivors'] for row in cases), 'required_survivors_when_positive': minimum,
            'mean_exact_survival': average([row['exact_survival'] for row in cases]),
            'mean_empirical_eval_survival': average([row['eval_survivors'] / plan['config']['mc_draws'] for row in cases])})
    rows = report['rows']
    means = [{'family': family, 'regime': regime, 'horizon': horizon,
              **{key: average([r[key] for r in rows if r['family'] == family and r['regime'] == regime and r['horizon'] == horizon])
                 for key in ('total_regret', 'information_advantage', 'approximation_regret')}}
             for family in FAMILIES for regime in REGIMES for horizon in (4, 8)]
    phase_times = {name: number(value['worker_seconds'], nonnegative=True) for name, value in closure['processes'].items()}
    return {'version': 'otto-cost-information-publication-v1', 'status': closure['status'], 'primary': groups,
        'support': support, 'all_policy_rows': rows, 'three_policy_means': means,
        'primary_case_records': report['gate']['cases'], 'h4_sampling_validation': report['h4_sampling_validation'],
        'config': plan['config'], 'attempted_prefixes': collection['cases'], 'retained_prefixes': collection['retained'],
        'prefix_exclusions': collection['prefix_exclusions'], 'worker_seconds': phase_times,
        'total_new_worker_seconds': math.fsum(phase_times.values()), 'prediction_times': prediction['prediction_times'],
        'collection_calls': collection['calls'], 'historical_parent_reference': closure['parent_reference'],
        'new_training': False, 'architecture_claim': False, 'case_selection': False,
        'interval_scope': 'Approximate hierarchical percentile intervals conditional on retained prefixes, realized selection streams and fixed policies; fit seeds are not independent data.',
        'h4_scope': 'Exact finite-grid law expectation at one fixed horizon, subject to recorded floating arithmetic.',
        'h8_scope': 'Evaluation-MC plug-in decomposition uses empirical survival mass and is descriptive, not an exact information floor.',
        'primary_scope': 'Separate selection draws pick the reference action; independent evaluation draws assess it on shared histories, including found as zero.'}


def render_figure(numbers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    for column, group in enumerate(numbers['primary']):
        regime, ax = group['regime'], axes[0, column]
        values = [group['control_gap'], group['reference_gap']]
        ax.bar([0, 1], values, color=['#2465a8', '#548b50'], width=.62)
        for x, value in enumerate(values):
            ax.annotate(f'{value:.5f}', (x, value), xytext=(0, 6), textcoords='offset points', ha='center')
        ax.set_xticks([0, 1], ['Frozen recurrent\nmean of 3 policies', 'Selection-stream\nreference action'])
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.)
        ax.set_title(f'{regime}: horizon-8 evaluation regret', loc='left', fontweight='bold')
        ax.set_ylabel('Mean unconditional teacher-cost regret')
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
        ax = axes[1, column]
        low, high = group['approximate_95_percent_interval'][2]
        estimate = group['gain_minus_required']
        ax.axvline(0., color='#555555', linestyle='--', linewidth=1)
        ax.hlines(0., low, high, color='#2465a8', linewidth=3)
        ax.plot([low, high], [0., 0.], '|', color='#2465a8', markersize=14)
        ax.scatter([estimate], [0.], color='#192e43', s=45, zorder=3)
        ax.set_yticks([])
        ax.set_ylim(-.8, .8)
        ax.margins(x=.22)
        ax.set_title(f"Gain minus {100 * group['required_fraction']:.0f}% of control regret", loc='left', fontweight='bold')
        ax.set_xlabel('Teacher-cost margin; positive favors the reference')
        ax.text(.5, .12, f'{estimate:.5f}  [{low:.5f}, {high:.5f}]\nApproximate 95% conditional percentile interval',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
    fig.suptitle(f"Fixed-horizon decision diagnostic: {numbers['status']}\n"
                 'Shared hypothetical histories | found draws count as zero | no new training', fontsize=14, fontweight='bold')
    fig.savefig(DEST / 'benchmark.png', dpi=160)
    plt.close(fig)


def document(numbers):
    lines = ['# Fixed-horizon decision diagnostic: ' + numbers['status'], '',
        ('**This measures decision headroom for frozen models, not a newly improved model.** '
        'All three original processes and the independent audit completed. The previous paired-action study remains '
        '`DEV_FAIL 6/18`. This diagnostic changes neither that result nor any training authorization.'), '',
        '![Frozen decisions, separate-stream reference and conditional intervals](otto-cost-information-results/benchmark.png)', '',
        ('At horizon 8, 128 selection draws choose one lowest-mean-cost reference action per public prefix. '
        'A separate 128-draw evaluation stream measures that fixed action and all frozen policies on the same histories. '
        'The control averages regrets of three `paired_recurrent` policies making separate decisions; it does not average '
        'their predicted costs into a new policy. Every retained prefix counts equally, and found draws contribute zero.'), '',
        '| Setting | Prefixes | Frozen recurrent regret | Reference regret | Gain | Required fraction | Gain minus requirement | Approx. 95% interval | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for group in numbers['primary']:
        lo, hi = group['approximate_95_percent_interval'][2]
        lines.append(f"| {group['regime']} | {group['cases']} | {group['control_gap']:.6f} | {group['reference_gap']:.6f} | "
                     f"{group['gain']:.6f} | {100 * group['required_fraction']:.0f}% | {group['gain_minus_required']:.6f} | "
                     f"[{lo:.6f}, {hi:.6f}] | {'pass' if group['passed'] else 'not resolved'} |")
    lines.extend(['', ('The criterion requires a positive lower margin endpoint, positive control regret, at least '
        f"{numbers['config']['min_cases']} retained prefixes per setting, and at least {numbers['config']['min_mc_survivors']} "
        'survivors in each stream for every prefix with positive exact survival probability. Conditions are retained separately:'), '',
        '| Setting | Minimum cases | MC support | Positive control regret | Positive lower margin |',
        '|---|---|---|---|---|'])
    for group in numbers['primary']:
        condition = group['conditions']
        lines.append('| ' + ' | '.join([group['regime'], *[str(condition[key]) for key in
            ('minimum_cases', 'mc_support', 'positive_control_gap', 'resolved_gain')]]) + ' |')
    lines.extend(['', ('Intervals are approximate hierarchical bootstrap percentiles, conditional on the retained '
        'prefixes, realized selection streams and fixed policies. They omit selection-algorithm variability. '
        'Fit seeds share evaluation data and are not independent datasets.'), '',
        '## Support and estimands', '',
        (f"Retained **{numbers['retained_prefixes']}/{numbers['attempted_prefixes']}** attempted prefixes; "
        f"**{numbers['prefix_exclusions']}** terminated within the eight observed analytic-policy transitions and were "
        'excluded without replacement. No actual native continuation follows the prefix.'), '',
        '| Setting | Retained | Exact zero-survival prefixes | Positive-survival prefixes below MC support | Min select/eval survivors | Mean exact survival | Mean evaluation-MC survival |',
        '|---|---:|---:|---:|---|---:|---:|'])
    for row in numbers['support']:
        lines.append(f"| {row['regime']} | {row['retained_prefixes']} | {row['exact_zero_survival_prefixes']} | "
                     f"{row['positive_survival_below_required_draws']} | {row['minimum_select_survivors']}/{row['minimum_eval_survivors']} | "
                     f"{row['mean_exact_survival']:.6f} | {row['mean_empirical_eval_survival']:.6f} |")
    lines.extend(['', ('Horizon 4 enumerates all 256 odor histories under the declared 53-bit root/sensor grid law. '
        'Its information and approximation terms are exact for that numerical reference, subject to recorded float arithmetic. '
        'Horizon 8 uses the evaluation stream for a descriptive plug-in decomposition, with its empirical survival denominator. '
        'The separately reported exact survival probability does not replace that denominator. The plug-in adaptive minimum '
        'is not an unbiased estimate or proof of a true information floor.'), '',
        ('For each prefix, unconditional regret is survival mass times conditional regret. These per-prefix values are '
        'averaged with zero-support prefixes retained as zero. Multiplying a conditional cohort mean by average survival mass '
        'would be a different quantity. The prior study\'s random average over surviving rows is also a different estimand.'), '',
        '## All twelve frozen policies', '',
        ('Every entry is unconditional mean teacher-cost regret. H4 is the finite-grid expectation; H8 is evaluation MC. '
        'All policy-level decomposition terms and the H4 sampling errors/estimated standard errors remain in the linked JSON.'), '',
        '| Frozen policy | Fit seed | H4 lambda3 | H4 lambda4 | H8 lambda3 | H8 lambda4 |',
        '|---|---|---:|---:|---:|---:|'])
    by = {(row['family'], row['fit_seed'], row['horizon'], row['regime']): row for row in numbers['all_policy_rows']}
    for family in FAMILIES:
        for seed in SEEDS:
            values = [by[family, seed, h, regime]['total_regret'] for h in (4, 8) for regime in REGIMES]
            lines.append('| ' + ' | '.join([LABELS[family], str(seed), *[f'{value:.6f}' for value in values]]) + ' |')
    lines.extend(['', '## Charged computation and interpretation', '', '| Original phase | Worker seconds |', '|---|---:|'])
    for name in ('collection', 'prediction', 'audit'):
        lines.append(f"| {name} | {numbers['worker_seconds'][name]:.3f} |")
    teacher = numbers['collection_calls']['teacher_score']['returned']
    native = numbers['collection_calls']['native_step']['returned']
    lines.extend(['', (f"Total new worker time: **{numbers['total_new_worker_seconds']:.3f}s**, including collection, "
        f"frozen inference/analysis and independent audit. Collection records **{teacher}** teacher annotations and "
        f"**{native}** actual prefix steps. Nested operation times must not be added to wall times. "
        'Qualification and publication are outside this total. Historical parent-model construction costs and all twelve '
        'inference timings are retained in the JSON; this is not a training-speed comparison.'), '',
        ('Publication clarification made before outcome inspection: the frozen protocol describes the reference as '
        'having richer information. The student actually reads all nine prefix rows, which retain the initial and '
        'eight later odor indicators, eight preceding-action indicators, discrete positions and sensing length. '
        'With the known law, that observed history may be reconstructible from the student inputs. This diagnostic '
        'therefore does not establish strictly extra observed information or irreducible input-compression loss.'), '',
        ('The reference instead has explicit belief reconstruction, the known sampling law, filter computation and '
        'teacher lookahead. Those computations carry costs and are not the learned 28-dimensional recurrent state. '
        'Information loss within that learned state is possible but is not established by this comparison. The '
        'reference receives no realized future observations when selecting its action. All costs are teacher labels, '
        'not demonstrated environment return.'), '',
        ('The audit independently checks public strict/legacy filtering, finite-grid weights, saved random draws, '
        'horizon-four lookup identities, scalar decomposition and bootstrap. It reconciles teacher costs with recorded '
        'forward masses and values without re-evaluating the teacher. No model is trained here, no policy runs autonomous '
        'continuations, and no architecture, biological-learning, robotics-transfer or novelty claim follows.'), '',
        ('[Complete scalars and support](otto-cost-information-results/summary.json) | '
        '[Protocol](otto-cost-information-protocol.md) | '
        '[Original closure](../output/otto-cost-information-v1/closure-01.json)'), ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-sha256', required=True)
    args = parser.parse_args()
    closure, report, prediction, collection, plan = authenticate(args.closure_sha256)
    require(not DEST.exists() and not DOCUMENT.exists(), 'exclusive first publication outputs')
    numbers = extract(closure, report, prediction, collection, plan)
    rendered = document(numbers)
    source = descriptor(Path(__file__))
    DEST.mkdir()
    render_figure(numbers)
    with DOCUMENT.open('x') as stream:
        stream.write(rendered)
    with (DEST / 'summary.json').open('x') as stream:
        json.dump(numbers, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    require(source == descriptor(Path(__file__)) and descriptor(STUDY / 'closure-01.json')['sha256'] == args.closure_sha256,
            'renderer and original closure unchanged')
    outputs = (DOCUMENT, DEST / 'summary.json', DEST / 'benchmark.png')
    receipt = {'scope': 'authenticated closed JSON publication only', 'closure': descriptor(STUDY / 'closure-01.json'),
        'audit': descriptor(STUDY / 'audit-01/audit.json'), 'renderer': source,
        'prediction_summary': descriptor(STUDY / 'prediction-01/summary.json'),
        'collection_summary': descriptor(STUDY / 'collection-01/summary.json'),
        'files': {str(path.relative_to(ROOT)): descriptor(path) for path in outputs},
        'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'native_calls': 0,
        'teacher_calls': 0, 'optimizer_calls': 0, 'case_selection': False}
    with (DEST / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': numbers['status'], 'report': str(DOCUMENT), 'figure': str(DEST / 'benchmark.png')}))


if __name__ == '__main__':
    main()
