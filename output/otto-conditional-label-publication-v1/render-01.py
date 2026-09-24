"""Publish only authenticated closed JSON from the conditional-label study.

No scientific arrays, checkpoints, journals, model or native runtime are read
as numerical content. Plotting starts only after all original process closures.
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
STUDY = ROOT / 'output/otto-conditional-label-v1'
DEST = ROOT / 'research/otto-conditional-label-results'
DOCUMENT = ROOT / 'research/otto-conditional-label-results.md'
FAMILIES = ('sampled', 'mean32')
SEEDS = (343000001, 343000002, 343000003)
REGIMES = ('lambda3', 'lambda4')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    require(path.is_file() and not path.is_symlink(), 'regular saved artifact')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def closed_json(directory, receipt, name):
    require(desc(directory / name) == receipt['files'][name], 'unchanged closed JSON ' + name)
    return read(directory / name)


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'finite publication scalar')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def authenticate(closure_sha):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{64}', closure_sha), 'repository root and external closure hash')
    closure_path, registration_path = STUDY / 'closure-01.json', STUDY / 'registration-01.json'
    require(desc(closure_path)['sha256'] == closure_sha, 'externally supplied original closure')
    closure = read(closure_path)
    require(closure['version'] == 'otto-conditional-label-v1' and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True and closure['new_training'] is True
            and closure['architecture_claim'] is False and closure['old_test_admitted'] is False
            and closure['new_execution_admitted'] is False and closure['registration'] == desc(registration_path)
            and set(closure['processes']) == {'collection', 'fit', 'audit'}, 'complete scoped study closure')
    sys.path.insert(0, str(ROOT / 'scripts'))
    import otto_conditional_label_common as c

    plan = read(registration_path)
    c.check_plan(plan)
    receipts, previous = {}, None
    for name, phase in (('collection', 'collect'), ('fit', 'fit'), ('audit', 'audit')):
        directory, terminal_path = STUDY / (name + '-01'), STUDY / (name + '-native-01.terminal.json')
        binding = closure['processes'][name]
        require(desc(directory / 'receipt.json') == binding['receipt'] and desc(terminal_path) == binding['terminal'], 'original phase byte joins')
        receipt, terminal = c.closed(directory, terminal_path), read(terminal_path)
        require(receipt['phase'] == phase and receipt['plan_sha256'] == closure['registration']['sha256']
                and receipt['wall_seconds'] == binding['worker_seconds']
                and receipt['old_test_decodes'] == receipt['astra_calls'] == 0, 'registered phase identity')
        require(previous is None or previous <= terminal['started_ns'], 'all original phases close sequentially')
        previous, receipts[name] = terminal['finished_ns'], receipt
    audit = closed_json(STUDY / 'audit-01', receipts['audit'], 'audit.json')
    fit = closed_json(STUDY / 'fit-01', receipts['fit'], 'summary.json')
    collection = closed_json(STUDY / 'collection-01', receipts['collection'], 'summary.json')
    normalization = closed_json(STUDY / 'fit-01', receipts['fit'], 'normalization.json')
    require(desc(STUDY / 'audit-01/audit.json') == closure['audit'] and audit['agreement'] is True
            and audit['technical_complete'] is False and audit['requires_original_supervisor_closure'] is True
            and audit['inputs']['collection_receipt'] == closure['processes']['collection']['receipt']
            and audit['inputs']['fit_receipt'] == closure['processes']['fit']['receipt'], 'independent original audit joins')
    report, gate = audit['report'], audit['gate']
    require(gate == report['gate'] == closure['gate'] and fit['status'] == gate['status'] == closure['status']
            and gate['status'] == ('DEV_PASS' if gate['passed'] else 'DEV_FAIL')
            and gate['architecture_claim'] is False and report['architecture_claim'] is False
            and report['normal_observation_claim'] is False and report['horizon'] == 8, 'unchanged prespecified outcome and scope')
    require(fit['collection_receipt'] == closure['processes']['collection']['receipt']
            and fit['calls'] == receipts['fit']['calls'] and collection['calls'] == receipts['collection']['calls']
            and collection['parent_reference'] == closure['parent_reference'] == plan['parent_reference'], 'same data lineage and counters')
    require(len(report['rows']) == 12 and {(r['family'], r['fit_seed'], r['regime']) for r in report['rows']}
            == {(f, s, regime) for f in FAMILIES for s in SEEDS for regime in REGIMES}, 'all six policies in both regimes')
    dev_count = fit['data_cases']['dev']
    require(len(report['cases']) == 6 * dev_count
            and len({(r['family'], r['fit_seed'], r['case_id']) for r in report['cases']}) == 6 * dev_count,
            'complete policy and case panels')
    require(len(fit['fits']) == len(fit['prediction_times']) == 6
            and {(r['family'], r['seed']) for r in fit['fits']} == {(f, s) for f in FAMILIES for s in SEEDS}
            and {(r['family'], r['seed']) for r in fit['prediction_times']} == {(f, s) for f in FAMILIES for s in SEEDS},
            'all six fit and inference times')
    require(len(gate['groups']) == 2 and [g['regime'] for g in gate['groups']] == list(REGIMES)
            and gate['passed'] is all(g['passed'] for g in gate['groups']), 'both required gate settings')
    for group in gate['groups']:
        require(group['cases'] == collection['counts']['dev_' + group['regime']]
                and group['required_fraction'] == plan['config']['required_fraction'] == .05
                and [r['fit_seed'] for r in group['paired_seed_gains']] == list(SEEDS)
                and group['interval_columns'] == ['sampled_gap', 'mean32_gap', 'gain', 'gain_minus_required']
                and len(group['approximate_95_percent_interval']) == 4
                and all(len(pair) == 2 and pair[0] <= pair[1] for pair in group['approximate_95_percent_interval'])
                and group['passed'] is all(group['conditions'].values()), 'complete paired gate and intervals')
    for value in (closure, audit, fit, collection, normalization):
        finite_tree(value)
    return closure, audit, fit, collection, normalization, plan


def extract(closure, audit, fit, collection, normalization, plan):
    report, support = audit['report'], []
    for regime in REGIMES:
        rows = [r for r in report['cases'] if r['family'] == FAMILIES[0]
                and r['fit_seed'] == SEEDS[0] and r['regime'] == regime]
        support.append({'regime': regime, 'retained_prefixes': len(rows),
            'all_found_banks': sum(r['alive_draws'] == 0 for r in rows),
            'surviving_draws': sum(r['alive_draws'] for r in rows),
            'allocated_draws': sum(r['draws'] for r in rows),
            'minimum_surviving_draws': min(r['alive_draws'] for r in rows)})
    worker = {key: value['worker_seconds'] for key, value in closure['processes'].items()}
    teachers = {split: sum(r['work']['teacher_calls'] for r in audit['probability_reconstruction'][split]['cases'])
                for split in ('train', 'dev')}
    require(sum(teachers.values()) == collection['calls']['teacher_score']['returned'], 'all charged teacher annotations')
    return {'version': 'otto-conditional-label-publication-v1', 'status': closure['status'],
        'gate': report['gate'], 'all_policy_rows': report['rows'], 'all_case_rows': report['cases'],
        'support': support, 'fits': fit['fits'], 'prediction_times': fit['prediction_times'],
        'worker_seconds': worker, 'total_new_worker_seconds': math.fsum(worker.values()),
        'teacher_endpoint_calls': teachers, 'collection_calls': collection['calls'], 'fit_calls': fit['calls'],
        'collection_counts': collection['counts'], 'attempted_prefixes': collection['cases'],
        'retained_prefixes': collection['retained'], 'prefix_exclusions': collection['prefix_exclusions'],
        'normalization': normalization, 'config': plan['config'], 'audit_limitations': audit['limitations'],
        'new_training': True, 'architecture_claim': False, 'normal_observation_claim': False,
        'autonomous_control_claim': False, 'case_selection': False,
        'interval_scope': 'Approximate hierarchical percentile intervals conditional on these fixed TRAIN banks and six fitted policies, not training uncertainty.',
        'intervention': 'One sampled label per update versus the empirical mean of the same32 labels; identical model, initialization, case order and optimizer-update budget.'}


def figure(numbers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    colors = ('#2864a4', '#c37718', '#548c52')
    for column, group in enumerate(numbers['gate']['groups']):
        ax = axes[0, column]
        values = []
        for paired, color in zip(group['paired_seed_gains'], colors, strict=True):
            y = [paired['sampled_gap'], paired['mean32_gap']]
            values.extend(y)
            ax.plot([0, 1], y, marker='o', color=color, label=str(paired['fit_seed']), alpha=.9)
        ax.set_xticks([0, 1], ['Sampled label', 'Mean of same 32'])
        ax.set_xlim(-.2, 1.2)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.)
        ax.set_ylabel('Mean unconditional teacher-cost regret')
        shift = 'TRAIN regime' if group['regime'] == 'lambda3' else 'Scenario shift'
        ax.set_title(f"{group['regime']}: {shift}", loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=.2)
        if column == 0:
            ax.legend(title='Paired fit seed', fontsize=8)
        ax = axes[1, column]
        estimate = group['gain_minus_required']
        low, high = group['approximate_95_percent_interval'][3]
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.hlines(0, low, high, linewidth=3, color='#2864a4')
        ax.plot([low, high], [0, 0], '|', markersize=14, color='#2864a4')
        ax.scatter([estimate], [0], s=45, color='#162c42', zorder=3)
        ax.set_yticks([])
        ax.set_ylim(-.8, .8)
        ax.margins(x=.22)
        ax.set_title('Pooled gain minus 5% of sampled regret', loc='left', fontweight='bold')
        ax.set_xlabel('Teacher-cost margin; positive favors mean32')
        ax.text(.5, .12, f'{estimate:.5f}  [{low:.5f}, {high:.5f}]\nApproximate 95% conditional percentile interval',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
    fig.suptitle('Same GRU, different target aggregation: ' + numbers['status']
                 + '\nHorizon 8 only | found costs remain zero | no architecture claim', fontsize=14, fontweight='bold')
    fig.savefig(DEST / 'benchmark.png', dpi=160)
    plt.close(fig)


def document(numbers):
    config = numbers['config']
    lines = ['# Same GRU, conditional cost labels: ' + numbers['status'], '',
        ('**This is a training-label comparison on a new fixed-horizon task.** Six models were fitted from scratch, '
         'and the independent saved-output audit completed under the original process closures. The result does not '
         'change earlier study outcomes or establish a new architecture.'), '',
        '![All paired fits and prespecified five-percent margins](otto-conditional-label-results/benchmark.png)', '',
        ('Both arms use the same 32 hypothetical teacher-cost labels per TRAIN prefix. `sampled` visits every label '
         'three times across 96 epochs; `mean32` uses their precomputed mean. Paired models have the same initialization, '
         'case order, optimizer recipe and number of updates. All eight blind transitions execute, but only the '
         'horizon-eight cost prediction is trained. Lambda3 is the TRAIN regime; lambda4 is a held-out sensing-regime shift.'), '',
        '| Setting | Prefixes | Sampled regret | Mean32 regret | Gain | Gain minus 5% | Approx. 95% margin interval | Gate |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for group in numbers['gate']['groups']:
        low, high = group['approximate_95_percent_interval'][3]
        lines.append(f"| {group['regime']} | {group['cases']} | {group['sampled_gap']:.6f} | {group['mean32_gap']:.6f} | "
                     f"{group['gain']:.6f} | {group['gain_minus_required']:.6f} | [{low:.6f}, {high:.6f}] | "
                     f"{'pass' if group['passed'] else 'fail'} |")
    lines.extend(['', ('Regret is `Q[chosen] - min(Q)`, using the lowest-index minimum predicted action. '
        'Found histories contribute zero and remain in every denominator. Draws are averaged within each prefix, '
        'then prefixes equally. The primary result averages three policies as separate decisions, not an ensemble of their costs.'), '',
        '| Setting | At least 64 prefixes | Positive sampled regret | Positive lower 5% margin | All 3 paired gains positive |',
        '|---|---|---|---|---|'])
    for group in numbers['gate']['groups']:
        condition = group['conditions']
        lines.append('| ' + ' | '.join([group['regime'], *[str(condition[key]) for key in
            ('minimum_cases', 'positive_sampled_gap', 'resolved_gain', 'all_paired_seeds_positive')]]) + ' |')
    lines.extend(['', ('Both settings must pass every condition. Intervals use 2,000 paired hierarchical bootstrap '
        'replicates per setting and are conditional on the fixed TRAIN bank and fitted policies. They do not include '
        'training variability or claim rigorous coverage. The 32/128 draws and three fit seeds are not independent cases.'), '',
        '## What the comparison controls', '',
        ('For each prefix, let `z_j` be centered teacher costs divided by 64, including found zeros, and '
         '`z_bar = mean_j(z_j)`. At a fixed prediction `f`, the finite-bank identity is'), '',
        '```text', 'mean_j ||f - z_j||^2 = ||f - z_bar||^2 + mean_j ||z_j - z_bar||^2.', '```', '',
        ('The last term does not depend on the prediction, so full-bank squared loss and mean-target loss have '
         'the same gradient and finite-bank optimum. The intervention here changes per-update gradient noise under '
         'matched optimizer work. It does not match target contributions per update, and a 32-draw mean is still an '
         'estimate rather than an exact conditional expectation. See the '
         '[target/loss implementation](../src/openjev/research/otto_conditional_label.py), '
         '[fabricated identity checks](../tests/test_otto_conditional_label.py) and '
         '[frozen protocol](otto-conditional-label-protocol.md).'), '',
        ('The common scale is derived only from all sampled TRAIN contrasts, not from the smaller mean-target '
         'variance. Both arms use the unchanged 28-state GRU: 8,299 total parameters, 8,096 potentially updated '
         'parameters and 203 frozen outcome/auxiliary-head parameters. Their forward compute is still charged. '
         'Only the public nine-row prefix, lengths and committed action sequence enter the model; beliefs, hidden '
         'sources, draws and teacher costs are not model inputs.'), '',
        '## Every paired fit and secondary loss', '',
        '| Setting | Fit seed | Sampled regret | Mean32 regret | Paired gain | Sampled target MSE: sampled / mean32 | Mean-target MSE: sampled / mean32 |',
        '|---|---|---:|---:|---:|---|---|'])
    by = {(r['family'], r['fit_seed'], r['regime']): r for r in numbers['all_policy_rows']}
    for group in numbers['gate']['groups']:
        for pair in group['paired_seed_gains']:
            a, b = (by[family, pair['fit_seed'], group['regime']] for family in FAMILIES)
            lines.append(f"| {group['regime']} | {pair['fit_seed']} | {pair['sampled_gap']:.6f} | {pair['mean32_gap']:.6f} | "
                f"{pair['gain']:.6f} | {a['sampled_target_mse']:.8f} / {b['sampled_target_mse']:.8f} | "
                f"{a['mean_target_mse']:.8f} / {b['mean_target_mse']:.8f} |")
    lines.extend(['', ('MSE is in unstandardized centered `Q/64` units against the independent DEV bank. '
        'Within-bank variance and all case-level scores remain in the JSON. Lower MSE alone does not imply lower decision regret.'), '',
        '## Support and charged computation', '',
        (f"Retained **{numbers['retained_prefixes']}/{numbers['attempted_prefixes']}** fresh observed prefixes; "
         f"**{numbers['prefix_exclusions']}** found-during-prefix exclusions were logged without replacement. "
         f"TRAIN retained **{numbers['collection_counts']['train']}** lambda3 prefixes with 32 draws each. "
         'No native environment continuation was executed after the observed prefix.'), '',
        '| DEV setting | Prefixes | All-found banks retained | Surviving / allocated draws | Minimum surviving draws |',
        '|---|---:|---:|---|---:|'])
    for row in numbers['support']:
        lines.append(f"| {row['regime']} | {row['retained_prefixes']} | {row['all_found_banks']} | "
                     f"{row['surviving_draws']}/{row['allocated_draws']} | {row['minimum_surviving_draws']} |")
    lines.extend(['', '| Original phase | Worker seconds |', '|---|---:|'])
    for name in ('collection', 'fit', 'audit'):
        lines.append(f"| {name} | {numbers['worker_seconds'][name]:.3f} |")
    lines.extend(['', '| Arm | Fit seed | Fit seconds | DEV inference seconds | Optimizer updates | Conceptual target contributions |',
                  '|---|---|---:|---:|---:|---:|'])
    timing = {(r['family'], r['seed']): r['seconds'] for r in numbers['prediction_times']}
    for row in numbers['fits']:
        lines.append(f"| {row['family']} | {row['seed']} | {row['seconds']:.3f} | {timing[row['family'], row['seed']]:.3f} | "
                     f"{row['updates']} | {row['target_draw_uses']} |")
    norm = numbers['normalization']
    lines.extend(['', (f"Total new worker time: **{numbers['total_new_worker_seconds']:.3f}s**. Shared bank construction "
        f"used **{numbers['teacher_endpoint_calls']['train']}** TRAIN and **{numbers['teacher_endpoint_calls']['dev']}** DEV "
        f"teacher endpoint annotations, plus **{numbers['collection_calls']['native_step']['returned']}** actual prefix steps. "
        f"The mean target was aggregated once from **{norm['mean_target_aggregation_draws']}** draws in "
        f"**{norm['mean_target_aggregation_seconds']:.6f}s**. Its repeated conceptual contributions are not repeated "
        'teacher calls or repeated reductions. Fit and inference times are nested within phase time; do not add '
        'them again. Qualification, historical teacher training and publication are outside this total. '
        'Matched updates do not establish equal wall time or a speedup.'), '',
        '## Interpretation limits', '',
        ('This is supervised fixed-path teacher-cost imitation. It differs from the earlier multi-horizon, '
        'survival-normalized objectives and leaves their gates unchanged. A pass supports this label-aggregation '
        'intervention under the registered recipe and scenario shift; a failure means it did not meet the preset '
        'continuation rule. Neither result establishes calibration, reinforcement learning, an architectural '
        'advance, autonomous return or benefit/no harm on normal-observation tasks. Those were not tested.'), '',
        ('The audit replays saved sampling laws, targets, schedules, metrics and bootstrap, and checks checkpoint '
        'schemas/hashes. Teacher values and optimizer updates are not independently re-executed. Paired initial '
        'equality remains a producer attestation supported by frozen source and fabricated invariants. All audit '
        'limitations are retained in the JSON.'), '',
        ('[Complete scores and costs](otto-conditional-label-results/summary.json) | '
        '[Protocol](otto-conditional-label-protocol.md) | '
        '[Original closure](../output/otto-conditional-label-v1/closure-01.json)'), ''])
    require(config['epochs'] == 96 and config['required_fraction'] == .05, 'fixed prose agrees with protocol')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-sha256', required=True)
    args = parser.parse_args()
    closure, audit, fit, collection, normalization, plan = authenticate(args.closure_sha256)
    require(not DEST.exists() and not DOCUMENT.exists(), 'exclusive first publication outputs')
    numbers = extract(closure, audit, fit, collection, normalization, plan)
    rendered = document(numbers)
    source = desc(Path(__file__))
    DEST.mkdir()
    figure(numbers)
    with DOCUMENT.open('x') as stream:
        stream.write(rendered)
    with (DEST / 'summary.json').open('x') as stream:
        json.dump(numbers, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    require(source == desc(Path(__file__)) and desc(STUDY / 'closure-01.json')['sha256'] == args.closure_sha256,
            'unchanged renderer and external closure')
    receipt = {'scope': 'authenticated closed JSON publication only', 'closure': desc(STUDY / 'closure-01.json'),
        'audit': desc(STUDY / 'audit-01/audit.json'), 'renderer': source,
        'fit_summary': desc(STUDY / 'fit-01/summary.json'), 'collection_summary': desc(STUDY / 'collection-01/summary.json'),
        'normalization': desc(STUDY / 'fit-01/normalization.json'),
        'files': {str(path.relative_to(ROOT)): desc(path) for path in (DOCUMENT, DEST / 'summary.json', DEST / 'benchmark.png')},
        'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'native_calls': 0,
        'teacher_calls': 0, 'optimizer_calls': 0, 'case_selection': False}
    with (DEST / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': numbers['status'], 'report': str(DOCUMENT), 'figure': str(DEST / 'benchmark.png')}))


if __name__ == '__main__':
    main()
