"""Render the closed action-gap pilot; saved JSON only, no models or data arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/otto-action-latent-v1'
DEST = ROOT / 'research/otto-action-latent-results'
FAMILIES = ('action_recurrent', 'action_blind', 'direct_horizon', 'ridge')
LABELS = ('Action recurrent', 'Action blind', 'Direct horizon', 'Solved ridge')
COLORS = ('#1765af', '#d7802f', '#59844b', '#8d659b')


def read(path):
    return json.loads(path.read_text())


def desc(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    closure = read(STUDY / 'closure-01.json')
    assert closure['technical_complete'] and closure['independent_audit_passed']
    for phase, row in closure['processes'].items():
        assert desc(STUDY / f'{phase}-01/receipt.json') == row['receipt']
        assert desc(STUDY / f'{phase}-native-01.terminal.json') == row['terminal']
    audit = read(STUDY / 'audit-01/audit.json')
    summary = read(STUDY / 'fit-01/summary.json')
    reports = audit['reports']
    gate = closure['gate']
    DEST.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for col, regime in enumerate(('lambda3', 'lambda4')):
        for row, (metric, label) in enumerate((('log_score', 'Outcome log loss'), ('decision_gap', 'Teacher-action cost gap'))):
            ax = axes[row, col]
            for family, name, color in zip(FAMILIES, LABELS, COLORS, strict=True):
                matches = [r for r in reports if r['family'] == family and r['condition'] == 'gap']
                ys = [[r['per_horizon'][str(h)]['by_regime'][regime]['case_weighted_' + metric] for h in range(1, 9)] for r in matches]
                mean = [sum(values) / len(values) for values in zip(*ys, strict=True)]
                ax.plot(range(1, 9), mean, marker='o', ms=4, lw=2, color=color, label=name)
                if family != 'ridge':
                    ax.fill_between(range(1, 9), [min(v) for v in zip(*ys, strict=True)], [max(v) for v in zip(*ys, strict=True)], color=color, alpha=.10, linewidth=0)
            ax.axvspan(.75, 4.5, color='#c7d4e0', alpha=.18)
            ax.axvline(4.5, color='#718398', linestyle='--', lw=1)
            ax.set(xlim=(.8, 8.2), ylim=(0, None), xticks=range(1, 9), xlabel='Unobserved action horizon', ylabel=label)
            ax.set_title(('Original sensing setting' if col == 0 else 'Sensing shift') + ' | ' + regime)
            ax.grid(axis='y', color='#dbe2e9', lw=.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Action-conditioned latent prediction: continuation rule failed (2/18)\nTrain gaps 1-4; evaluate 1-8 | lower is better', fontsize=16, fontweight='bold')
    fig.savefig(DEST / 'horizons.png', dpi=160)
    plt.close(fig)
    lines = ['# Every action-gap result', '', 'All values below are independently reconstructed from the saved final predictions.',
             'No model, teacher or native environment was called to create this report.', '',
             'Means below average three fit seeds over the same cases; ridge repeats one deterministic fit.',
             'Decision gap averages surviving targets within each case, then cases with support.', '',
             '| Method | Setting | Long-gap log loss | Long-gap decision gap | Normal log loss | Normal decision gap |',
             '|---|---|---:|---:|---:|---:|']
    means = {}
    for family, label in zip(FAMILIES, LABELS, strict=True):
        for regime in ('lambda3', 'lambda4'):
            values = []
            for condition, group in (('gap', 'long'), ('normal', 'all')):
                rows = [r['groups'][group]['by_regime'][regime] for r in reports if r['family'] == family and r['condition'] == condition]
                values.extend(sum(r['case_weighted_' + m] for r in rows) / len(rows) for m in ('log_score', 'decision_gap'))
            means[family, regime] = values
            lines.append('| ' + ' | '.join([label, regime, *[f'{v:.6f}' for v in values]]) + ' |')
    lines.extend(['', '## Long-gap Brier score', '', 'This second proper score is retained even when its ranking differs from log loss.', '', '| Method | lambda3 | lambda4 |', '|---|---:|---:|'])
    for family, label in zip(FAMILIES, LABELS, strict=True):
        values = [sum(r['groups']['long']['by_regime'][regime]['case_weighted_brier'] for r in reports if r['family'] == family and r['condition'] == 'gap') / 3 for regime in ('lambda3', 'lambda4')]
        lines.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    lines.extend(['', '## Every fit and comparison', '', '| Seed | Setting | Control | Long log loss | Long decision gap | Normal log loss | Normal decision gap | Supported cases | Pass |', '|---|---|---|---|---|---|---|---|---|'])
    for cell in gate['cells']:
        checks = {r['name']: r for r in cell['conditions']}
        values = []
        for condition in ('gap', 'normal'):
            for metric in ('log_score', 'decision_gap'):
                r = checks[condition + '_' + metric]
                values.append(f"{r['candidate']:.6f} / {r['control']:.6f} ({'pass' if r['passed'] else 'fail'})")
        lines.append('| ' + ' | '.join([str(cell['fit_seed']), cell['regime'], cell['control'], *values,
                    f"{checks['gap_support']['actual']} long; {checks['normal_support']['actual']} normal", str(cell['passed'])]) + ' |')
    lines.extend(['', 'Each metric cell shows candidate / control. All six conditions per cell must pass.', '', '## Costs', '',
                  '| Method | Seed | Fit seconds | Updates | Parameters |', '|---|---|---:|---:|---:|'])
    for row in summary['fits']:
        lines.append(f"| {row['family']} | {row['seed']} | {row['seconds']:.6f} | {row['updates']} | {row['parameters']['count']} |")
    lines.extend(['', f"Ridge: one fit, {summary['ridge_fit']['seconds']:.6f} seconds, 3,024 coefficients.", '',
                  '| Method | Seed | Panel | Cases | Inference seconds |', '|---|---|---|---:|---:|'])
    for row in summary['prediction_times']:
        lines.append(f"| {row['family']} | {row['seed']} | {row['condition']} | {row['cases']} | {row['seconds']:.6f} |")
    lines.extend(['', 'Fit times include construction, optimization, final parameter checks and checkpoint writing. The whole producer additionally includes data preparation, scoring, all prediction views and publication of its artifacts.',
                  'Inference is batch one, with Python validation and feature/input preparation, excluding model construction/loading. No end-to-end serving or hardware-independent speed claim.', '',
                  '## Support and interpretation', ''])
    for regime in ('lambda3', 'lambda4'):
        r = next(r for r in reports if r['family'] == 'action_recurrent' and r['condition'] == 'gap')
        for group in ('short', 'long', 'all'):
            v = r['groups'][group]['by_regime'][regime]
            lines.append(f"* {regime}, {group}: {v['declared_cases']} originating cases, {v['supported_cases']} with decision support, {v['decision_rows']}/{v['outcome_rows']} nonterminal targets.")
    lines.extend(['', 'Shading shows the range across three fits, not a confidence interval. All three fits share the same evaluation cases. The absorbing-found suffix is scored for outcomes and excluded from decisions.',
                  'Normal forecasts keep finite learned logits after observed found while freezing state. They do not take the available exact terminal-probability shortcut; this is an implementation limitation shared by all controls.', '',
                  '[Protocol](../otto-action-latent-protocol.md) | [Interpretation](../otto-action-latent-results.md) | [Complete evidence and checkpoints](https://github.com/kw2828/OpenJev/releases/tag/otto-action-latent-v1)', ''])
    (DEST / 'README.md').write_text('\n'.join(lines))
    receipt = {'scope': 'closed JSON rendering only', 'closure': desc(STUDY / 'closure-01.json'),
               'audit': desc(STUDY / 'audit-01/audit.json'), 'summary': desc(STUDY / 'fit-01/summary.json'),
               'renderer': desc(Path(__file__)), 'files': {p.name: desc(p) for p in DEST.iterdir()},
               'new_model_calls': 0, 'new_native_calls': 0, 'array_decodes': 0}
    (DEST / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'means': {f'{f}:{r}': v for (f, r), v in means.items()}, 'gate': [gate['passed_cells'], gate['total_cells']]}))


if __name__ == '__main__':
    main()
