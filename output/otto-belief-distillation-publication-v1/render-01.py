"""Publish a closed belief-distillation study from authenticated saved JSON only.

Authored before closure. Execution requires the externally supplied final closure
hash. It never imports study code, loads an array/checkpoint, or calls a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'output/otto-belief-distillation-v1'
DEST = ROOT / 'research/otto-belief-distillation-results'
REPORT = ROOT / 'research/otto-belief-distillation-results.md'
FAMILIES = ('recurrent_soft', 'recurrent_sampled', 'action_blind_soft', 'direct_soft')
LABELS = ('Recurrent + soft', 'Recurrent + sampled', 'Action blind + soft', 'Direct + soft')
COLORS = ('#1765af', '#d7802f', '#59844b', '#8d659b')
SEEDS = (327000001, 327000002, 327000003)
REGIMES = ('lambda3', 'lambda4')
METRICS = ('sampled_nll', 'decision_gap', 'oracle_kl', 'action_effect_error')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def desc(path):
    require(path.is_file() and not path.is_symlink(), 'regular evidence file: ' + str(path))
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative saved metric')
    return float(value)


def mean(values):
    return math.fsum(values) / len(values)


def percentage(candidate, control):
    return None if control == 0 else 100 * (control - candidate) / control


def pct(value):
    return 'undefined (zero control)' if value is None else f'{value:+.2f}%'


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def authenticated_json(directory, receipt, name):
    path = directory / name
    require(desc(path) == receipt['files'][name], 'closed JSON identity: ' + name)
    return read(path)


def authenticate(sha):
    closure_path = STUDY / 'closure-01.json'
    require(desc(closure_path)['sha256'] == sha, 'externally supplied closure pin')
    closure = read(closure_path)
    require(closure['version'] == 'otto-belief-distillation-v1'
            and closure['technical_complete'] is True and closure['independent_audit_passed'] is True
            and closure['status'] in ('DEV_PASS', 'DEV_FAIL'), 'closed successful original processes')
    require(set(closure['processes']) == {'collection', 'fit', 'audit'}, 'all original phases')
    receipts, terminals = {}, {}
    for phase, row in closure['processes'].items():
        directory = STUDY / f'{phase}-01'
        receipt_path = directory / 'receipt.json'
        terminal_path = STUDY / f'{phase}-native-01.terminal.json'
        require(desc(receipt_path) == row['receipt'] and desc(terminal_path) == row['terminal'], 'original closure joins')
        receipt, terminal = read(receipt_path), read(terminal_path)
        start = authenticated_json(directory, receipt, 'started.json')
        launch = start['launch']
        require(receipt['status'] == terminal['status'] == 'completed' and terminal['returncode'] == 0
                and terminal['group_absent'] is True and terminal['timed_out'] is False
                and terminal['error'] is None and terminal['clock_error'] is None
                and terminal['timing_available'] is True and terminal['cleanup']['reaped'] is True
                and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == []
                and all(terminal[k] == v for k, v in launch.items())
                and launch['started_ns'] <= receipt['started_ns'] < receipt['finished_ns'] <= terminal['finished_ns'] <= terminal['deadline_ns']
                and terminal['deadline_ns'] == terminal['started_ns'] + terminal['cap_seconds'] * 10**9,
                'genuine bounded original process closure')
        launch_path = Path(terminal['command'][terminal['command'].index('--supervision') + 1])
        require(desc(launch_path) == receipt['supervision'], 'original launch bytes')
        require(receipt['plan_sha256'] == start['plan_sha256'], 'original registration identity')
        receipts[phase], terminals[phase] = receipt, terminal
    require(len({r['plan_sha256'] for r in receipts.values()}) == 1
            and terminals['collection']['finished_ns'] <= terminals['fit']['started_ns']
            and terminals['fit']['finished_ns'] <= terminals['audit']['started_ns'], 'sequential original phase joins')
    audit = authenticated_json(STUDY / 'audit-01', receipts['audit'], 'audit.json')
    summary = authenticated_json(STUDY / 'fit-01', receipts['fit'], 'summary.json')
    collection = authenticated_json(STUDY / 'collection-01', receipts['collection'], 'summary.json')
    require(audit['agreement'] is True and audit['requires_original_supervisor_closure'] is True
            and audit['gate'] == closure['gate'] and summary['status'] == closure['status'], 'independent gate agrees')
    require(audit['inputs']['collection_receipt'] == closure['processes']['collection']['receipt']
            and audit['inputs']['fit_receipt'] == closure['processes']['fit']['receipt'], 'audit binds same original producers')
    gate = audit['gate']
    require(gate['candidate'] == FAMILIES[0] and gate['controls'] == list(FAMILIES[1:])
            and gate['fit_seeds'] == list(SEEDS) and gate['regimes'] == list(REGIMES)
            and gate['total_cells'] == len(gate['cells']) == 18
            and gate['passed_cells'] == sum(cell['passed'] for cell in gate['cells'])
            and gate['passed'] is all(cell['passed'] for cell in gate['cells'])
            and closure['status'] == ('DEV_PASS' if gate['passed'] else 'DEV_FAIL'), 'fixed complete gate')
    reports = {(r['family'], r['fit_seed'], r['condition']): r for r in audit['reports']}
    effects = {(r['family'], r['fit_seed']): r['report'] for r in audit['sensitivity']}
    require(len(audit['reports']) == len(reports) == 24
            and set(reports) == {(f, s, p) for f in FAMILIES for s in SEEDS for p in ('gap', 'normal')}
            and len(audit['sensitivity']) == len(effects) == 12
            and set(effects) == {(f, s) for f in FAMILIES for s in SEEDS}, 'complete paired results')
    return closure, audit, summary, collection, reports, effects


def stats(reports, effects):
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            gap, normal = reports[family, seed, 'gap'], reports[family, seed, 'normal']
            for regime in REGIMES:
                long = gap['groups']['long']['by_regime'][regime]
                ordinary = normal['groups']['all']['by_regime'][regime]
                oracle = gap['oracle']['groups']['long']['all_rows']['by_regime'][regime]
                effect = [effects[family, seed]['per_horizon'][str(h)]['by_regime'][regime] for h in range(5, 9)]
                rows.append({'family': family, 'fit_seed': seed, 'regime': regime,
                    'sampled_nll': finite(long['case_weighted_log_score']),
                    'decision_gap': finite(long['case_weighted_decision_gap']),
                    'oracle_kl': finite(oracle['case_weighted_kl']),
                    'action_effect_error': mean([finite(r['case_weighted_model_effect_error']) for r in effect]),
                    'oracle_signal': mean([finite(r['case_weighted_oracle_signal']) for r in effect]),
                    'normal_nll': finite(ordinary['case_weighted_log_score']),
                    'normal_decision_gap': finite(ordinary['case_weighted_decision_gap']),
                    'declared_cases': long['declared_cases'], 'supported_cases': long['supported_cases'],
                    'decision_rows': long['decision_rows'], 'outcome_rows': long['outcome_rows']})
    fields = (*METRICS, 'oracle_signal', 'normal_nll', 'normal_decision_gap')
    aggregate = [{'family': f, 'regime': r, **{k: mean([x[k] for x in rows if x['family'] == f and x['regime'] == r])
                  for k in fields}} for f in FAMILIES for r in REGIMES]
    by = {(r['family'], r['fit_seed'], r['regime']): r for r in rows}
    comparisons = [{'control': control, 'fit_seed': seed, 'regime': regime,
        **{k: percentage(by[FAMILIES[0], seed, regime][k], by[control, seed, regime][k])
           for k in (*METRICS, 'normal_nll', 'normal_decision_gap')}}
        for control in FAMILIES[1:] for regime in REGIMES for seed in SEEDS]
    return {'per_seed': rows, 'means': aggregate, 'paired_relative_reductions_pct': comparisons,
            'averaging': 'equal fit seeds over the same cases; action-effect E averages horizons5-8, all cases retained',
            'negative_percentage_means_candidate_is_worse': True}


def figure(numbers, gate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), layout='constrained')
    titles = ('Sampled outcome log loss', 'Teacher-action cost gap', 'KL from full-belief target', 'Action-effect error E')
    rows = numbers['per_seed']
    for ax, metric, title in zip(axes.flat, METRICS, titles, strict=True):
        for j, (family, label, color) in enumerate(zip(FAMILIES, LABELS, COLORS, strict=True)):
            for i, regime in enumerate(REGIMES):
                values = [r[metric] for r in rows if r['family'] == family and r['regime'] == regime]
                x = i * 1.4 + (j - 1.5) * .21
                ax.bar(x, mean(values), width=.18, color=color, alpha=.82, label=label if i == 0 else None)
                for k, value in enumerate(values):
                    ax.scatter(x + (k - 1) * .035, value, s=24, marker=('o', '^', 's')[k],
                               facecolor='white', edgecolor='#26323d', linewidth=.8, zorder=3)
        if metric == 'action_effect_error':
            for i, regime in enumerate(REGIMES):
                signal = next(r['oracle_signal'] for r in rows if r['regime'] == regime)
                ax.hlines(signal, i * 1.4 - .43, i * 1.4 + .43, color='#253746', linestyle=':', lw=1.4)
            ax.text(.02, .96, 'Dotted: oracle action signal S', transform=ax.transAxes, va='top', fontsize=9)
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xticks((0, 1.4), ('lambda3 | training setting', 'lambda4 | sensing shift'))
        ax.set_ylim(bottom=0)
        ax.set_xlim(-.58, 1.98)
        ax.grid(axis='y', color='#dbe2e9', lw=.7, zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=10)
    state = 'passed' if gate['passed'] else 'failed'
    fig.suptitle(f"Full-belief targets for compact prediction: DEV rule {state} ({gate['passed_cells']}/18)\n"
                 'Unobserved horizons 5-8 | all metrics lower is better | bars: three-fit means; dots: individual fits',
                 fontsize=15, fontweight='bold')
    fig.savefig(DEST / 'benchmark.png', dpi=160)
    fig.savefig(DEST / 'benchmark.svg')
    plt.close(fig)


def documents(closure, audit, summary, collection, numbers):
    gate = closure['gate']
    sampled = [r for r in numbers['paired_relative_reductions_pct'] if r['control'] == 'recurrent_sampled']
    sampled_nll = {regime: mean([r['sampled_nll'] for r in sampled if r['regime'] == regime]) for regime in REGIMES}
    nll_wins = sum(r['sampled_nll'] > 0 for r in sampled)
    gap_wins = sum(r['decision_gap'] > 0 for r in sampled)
    shifted = {r['family']: r for r in numbers['means'] if r['regime'] == 'lambda4'}
    direct_shift = [r for r in numbers['paired_relative_reductions_pct']
                    if r['control'] == 'direct_soft' and r['regime'] == 'lambda4']
    effect_losses = sum(r['action_effect_error'] < 0 for r in direct_shift)
    title = ('Full-belief distillation passed the development screen' if gate['passed']
             else 'Full-belief distillation did not pass the development screen')
    lines = ['# ' + title, '', f"**{closure['status']}: {gate['passed_cells']} of 18 paired comparisons passed.** "
             'All 12 prescribed fits and the independent saved-output audit completed. '
             'The original continuation rule is unchanged; oracle diagnostics do not replace it.', '',
             f"The strongest result is probability supervision: recurrent soft lowers long-gap log loss against "
             f"its sampled-label twin in **{nll_wins}/6 seed/regime pairs**, with mean paired reductions of "
             f"**{sampled_nll['lambda3']:.2f}% / {sampled_nll['lambda4']:.2f}%** in lambda3/lambda4. "
             f"Long-gap decision cost improves in only **{gap_wins}/6** of those pairs. "
             'Better probability forecasts did not produce a consistent decision advantage.', '',
             f"The action-effect diagnostic is also mixed. Under the sensing shift, recurrent soft has mean "
             f"E=**{shifted['recurrent_soft']['action_effect_error']:.6f}**, versus "
             f"**{shifted['action_blind_soft']['action_effect_error']:.6f}** for action blind and "
             f"**{shifted['direct_soft']['action_effect_error']:.6f}** for direct. "
             f"It is worse than direct in **{effect_losses}/3 seeds**. This measures error in the direction and "
             'size of the probability change under opposite actions, not merely sensitivity to action input.', '',
             '![Four matched methods: long-gap predictive and action-effect errors](otto-belief-distillation-results/benchmark.png)', '',
             'Bars average three fit seeds on the same fresh development cases; dots show each fit. '
             'These are not confidence intervals or independent datasets. The bottom panels are supplementary diagnostics.', '',
             'The experiment changes categorical supervision while keeping the 28-state model, inputs, optimizer and data matched. '
             'Recurrent soft versus recurrent sampled isolates the target distribution within this study. '
             'Action blind and direct use the same soft targets. The full-belief teacher has privileged information '
             'that never enters neural inference.', '',
             '## Long-gap results', '',
             '| Method | Setting | Sampled NLL | Decision gap | Oracle KL | Action-effect E |',
             '|---|---|---:|---:|---:|---:|']
    names = dict(zip(FAMILIES, LABELS, strict=True))
    for row in numbers['means']:
        lines.append('| ' + ' | '.join([names[row['family']], row['regime'], *[f'{row[k]:.6f}' for k in METRICS]]) + ' |')
    lines.extend(['', 'Every value above covers unobserved horizons 5-8. NLL is sampled negative log likelihood. '
                  'NLL and KL include absorbing-found suffixes. '
                  'Decision gap averages nonterminal rows within each case, then supported cases. '
                  'Action-effect E retains every case and measures error in the predicted change under action XOR1. '
                  'Its dotted reference S is the full-belief action signal; E=S for an exactly action-blind predictor.', '',
                  '## Paired effects', '',
                  'Positive percentages favor recurrent soft. Entries are means of the three paired reductions '
                  '`100*(control-candidate)/control`, followed by their minimum and maximum. '
                  'They are not ratios of the displayed means.', '',
                  '| Control | Setting | NLL reduction | Decision-gap reduction | KL reduction | Effect-error reduction |',
                  '|---|---|---:|---:|---:|---:|'])
    for control in FAMILIES[1:]:
        for regime in REGIMES:
            rows = [r for r in numbers['paired_relative_reductions_pct'] if r['control'] == control and r['regime'] == regime]
            values = []
            for metric in METRICS:
                numbers_here = [r[metric] for r in rows]
                values.append('undefined' if any(v is None for v in numbers_here)
                              else f'{pct(mean(numbers_here))} [{pct(min(numbers_here))}, {pct(max(numbers_here))}]')
            lines.append('| ' + ' | '.join([names[control], regime, *values]) + ' |')
    lines.extend(['', 'The registered gate requires at least 1% lower sampled NLL and 5% lower decision gap '
                  'at long horizons against each control for all three seeds and both regimes, '
                  'no more than 1% normal-panel regression, and at least 48 supported cases. '
                  'A favorable average cannot rescue a failed cell.', '',
                  '## Data, computation and limits', ''])
    counts = collection['counts']
    lines.append(f"Collection retained **{counts['train']['lambda3']}/1536 TRAIN**, "
                 f"**{counts['dev']['lambda3']}/128 lambda3 DEV** and **{counts['dev']['lambda4']}/128 lambda4 DEV** cases. "
                 f"The {collection['prefix_exclusions']} prefix terminations were excluded without replacement. "
                 'This estimates performance conditional on surviving the fixed analytic prefix.')
    lines.append('')
    for regime in REGIMES:
        row = next(r for r in numbers['per_seed'] if r['family'] == FAMILIES[0] and r['regime'] == regime)
        lines.append(f"* {regime}, long gaps: {row['declared_cases']} originating cases; {row['supported_cases']} "
                     f"with decision support; {row['decision_rows']}/{row['outcome_rows']} nonterminal target rows.")
    lines.extend(['', '| Method | Fit time mean | Fit time range |', '|---|---:|---:|'])
    for family in FAMILIES:
        times = [finite(r['seconds']) for r in summary['fits'] if r['family'] == family]
        require(len(times) == 3, 'three recorded fit costs')
        lines.append(f"| {names[family]} | {mean(times):.3f}s | {min(times):.3f}-{max(times):.3f}s |")
    lines.append('')
    lines.append('Original process worker times: ' + '; '.join(f"{phase} {row['worker_seconds']:.3f}s"
                 for phase, row in closure['processes'].items()) + '.')
    calls = collection['calls']
    lines.append(f"Collection charged {calls['native_step']['returned']} native steps, "
                 f"{calls['teacher_score']['returned']} teacher annotations, {calls['sampler_table']['returned']} sensor tables, "
                 f"{calls['sampler_lookup']['returned']} likelihood lookups and {calls['shadow_update']['returned']} Bayesian updates. "
                 'Nested call times cannot be added. These are implementation-specific costs, not a general speedup claim.')
    lines.extend(['', 'Normal forecasts use an exact terminal shortcut only after found was observed. Gap forecasts receive '
                  'no future observations or found flags. The opposite-action diagnostic invents no alternate observations.', '',
                  'The full-belief reference uses the authenticated scalar sensor kernel and declared 53-bit uniform-grid law. '
                  'It is a privileged probabilistic reference, not a fair learned baseline or deterministic PRNG replay. '
                  'The audit reconstructs DEV beliefs, targets and metrics; it authenticates TRAIN variance and raw sensor '
                  'primitive values as records rather than replaying training or native code.', '',
                  'This is a development forecasting study, not autonomous search, RL, connectome evidence, robotics transfer '
                  'or architectural novelty. Earlier TEST and confirmation panels remain closed. '
                  'The larger dataset and shared terminal shortcut also prevent attributing cross-study changes solely to distillation.', '',
                  'The 7/18 result cannot be treated as progress over the previous pilot\'s 2/18: '
                  'the case allocation and terminal treatment changed. The matched comparisons here are the evidence.', '',
                  '[Every seed, normal-panel result and cost](otto-belief-distillation-results/README.md) | '
                  '[Protocol](otto-belief-distillation-protocol.md) | '
                  '[Closed evidence](../output/otto-belief-distillation-v1/closure-01.json)', ''])
    appendix = ['# Every belief-distillation comparison', '',
                'Saved independent-audit scalars only. Positive reductions favor recurrent soft. '
                'Fit seeds share the same development cases. No case-level data is duplicated here.', '',
                '| Seed | Setting | Control | NLL reduction | Decision-gap reduction | KL reduction | Effect-error reduction | Normal NLL reduction | Normal gap reduction |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in numbers['paired_relative_reductions_pct']:
        appendix.append('| ' + ' | '.join([str(row['fit_seed']), row['regime'], names[row['control']],
            *[pct(row[k]) for k in (*METRICS, 'normal_nll', 'normal_decision_gap')]]) + ' |')
    appendix.extend(['', '## Original gate', '', '| Seed | Setting | Control | Passed conditions | Cell passes |',
                     '|---|---|---|---|---|'])
    for cell in gate['cells']:
        passed = ', '.join(r['name'] for r in cell['conditions'] if r['passed']) or 'none'
        appendix.append(f"| {cell['fit_seed']} | {cell['regime']} | {cell['control']} | {passed} | {cell['passed']} |")
    appendix.extend(['', '## All method and seed scalars', '',
                     '| Method | Seed | Setting | Long NLL | Long gap | Long KL | Long effect E | Normal NLL | Normal gap |',
                     '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for row in numbers['per_seed']:
        appendix.append('| ' + ' | '.join([names[row['family']], str(row['fit_seed']), row['regime'],
            *[f'{row[k]:.9f}' for k in (*METRICS, 'normal_nll', 'normal_decision_gap')]]) + ' |')
    appendix.extend(['', '## Recorded fit and inference costs', '', '| Method | Seed | Fit seconds | Updates | Exposures |',
                     '|---|---|---:|---:|---:|'])
    for row in summary['fits']:
        appendix.append(f"| {row['family']} | {row['seed']} | {row['seconds']:.6f} | {row['updates']} | {row['exposures']} |")
    appendix.extend(['', '| Method | Seed | Panel | Cases | Inference seconds |', '|---|---|---|---:|---:|'])
    for row in summary['prediction_times']:
        appendix.append(f"| {row['family']} | {row['seed']} | {row['condition']} | {row['cases']} | {row['seconds']:.6f} |")
    appendix.extend(['', '[Main interpretation](../otto-belief-distillation-results.md) | [Compact scalar JSON](summary.json)', ''])
    return '\n'.join(lines), '\n'.join(appendix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure-sha256', required=True)
    args = parser.parse_args()
    closure, audit, summary, collection, reports, effects = authenticate(args.closure_sha256)
    require(not DEST.exists() and not REPORT.exists(), 'exclusive new publication outputs')
    numbers = stats(reports, effects)
    main_text, appendix = documents(closure, audit, summary, collection, numbers)
    DEST.mkdir()
    figure(numbers, closure['gate'])
    with REPORT.open('x') as stream:
        stream.write(main_text)
    with (DEST / 'README.md').open('x') as stream:
        stream.write(appendix)
    write_json(DEST / 'summary.json', numbers)
    receipt = {'scope': 'closed saved JSON rendering only', 'closure': desc(STUDY / 'closure-01.json'),
               'audit': desc(STUDY / 'audit-01/audit.json'), 'fit_summary': desc(STUDY / 'fit-01/summary.json'),
               'collection_summary': desc(STUDY / 'collection-01/summary.json'), 'renderer': desc(Path(__file__)),
               'files': {str(p.relative_to(ROOT)): desc(p) for p in [REPORT, *DEST.iterdir()]},
               'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'native_calls': 0,
               'teacher_calls': 0, 'optimizer_calls': 0, 'case_selection': False}
    write_json(DEST / 'receipt.json', receipt)
    print(json.dumps({'status': closure['status'], 'gate': [closure['gate']['passed_cells'], 18],
                      'figure': str(DEST / 'benchmark.png'), 'report': str(REPORT)}))


if __name__ == '__main__':
    main()
