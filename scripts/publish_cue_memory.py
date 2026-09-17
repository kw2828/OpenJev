"""Publish verified cue-visible adapted MiniGrid results without running models."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import publish_recurrent_world as base
from matplotlib.patches import Patch

VERSION = 'cue-memory-v1'
MODES = ('cue_swapped', 'native_start')
STEMS = ('cue-memory-results', 'cue-memory-learning', 'cue-memory-native-start')
RECEIPT = 'cue-memory-figures.json'
SHORT_LABELS = ['Current\nPPO', 'Recurrent\nPPO', '+ reward', '+ full', 'Random']


def load_diagnostics(report, source, plan_path, summary, training):
    """Replay every diagnostic aggregate from its matched per-episode records."""
    path = report / 'diagnostic.json'
    diagnostic = json.loads(path.read_text())
    p = summary['protocol']
    if (diagnostic['protocol_version'] != VERSION or diagnostic['plan_sha256'] != base.sha(plan_path)
            or diagnostic['summary_sha256'] != base.sha(source)):
        raise ValueError('Diagnostic report must match this cue-visible summary and frozen plan')
    if diagnostic['novelty_established'] is not False or diagnostic['independent_confirmation'] is not False:
        raise ValueError('This development publisher cannot establish novelty or independent confirmation')
    expected = {(arm, seed, mode) for arm in base.ARMS for seed in p['seeds'] for mode in MODES}
    observed, scores, aggregates, paired_rows = set(), {}, {}, {}
    for record in diagnostic['evaluations']:
        arm, seed, mode = record['arm'], record['seed'], record['mode']
        identity = (arm, seed, mode)
        if identity not in expected or identity in observed:
            raise ValueError('Unexpected or duplicated diagnostic condition')
        observed.add(identity)
        if record['checkpoint_sha256'] != training[(arm, seed)]['checkpoint_sha256']:
            raise ValueError('Diagnostic used a different trained checkpoint')
        results = record['results']
        if len(results) != len(p['eval_sizes']) or {r['size'] for r in results} != set(p['eval_sizes']):
            raise ValueError('Diagnostic map size coverage mismatch')
        for result in results:
            size, rows = result['size'], result['episodes']
            first = p['evaluation_seed_start'] + size * 10000
            if [row['seed'] for row in rows] != list(range(first, first + p['evaluation_episodes'])):
                raise ValueError('Diagnostic episode seeds do not match primary evaluation')
            for row in rows:
                if (any(type(row[key]) is not bool for key in ('success', 'wrong_goal', 'timeout'))
                        or sum(row[key] for key in ('success', 'wrong_goal', 'timeout')) != 1):
                    raise ValueError('Invalid diagnostic terminal outcome partition')
                if (not isinstance(row['length'], int) or not 1 <= row['length'] <= p['max_steps']
                        or not np.isfinite(row['return'])):
                    raise ValueError('Invalid diagnostic episode length or return')
            fields = [('success', 'success'), ('wrong_goal', 'wrong_goal'), ('timeout', 'timeout'),
                      ('mean_return', 'return'), ('mean_length', 'length')]
            for key, field in fields:
                base.close(result[key], np.mean([row[field] for row in rows]), f'diagnostic {key}')
            scores[(arm, seed, mode, size)] = result['success']
            paired_rows[(arm, seed, mode, size)] = rows
            aggregates[(arm, seed, mode, size)] = {key: result[key] for key, _ in fields}
    if observed != expected:
        raise ValueError('Diagnostic conditions are incomplete')
    averages = diagnostic['averages']
    if set(averages) != set(MODES):
        raise ValueError('Unexpected diagnostic average modes')
    for mode in MODES:
        if set(averages[mode]) != set(base.ARMS):
            raise ValueError('Missing diagnostic average arms')
        for arm in base.ARMS:
            if set(averages[mode][arm]) != {str(s) for s in p['eval_sizes']}:
                raise ValueError('Missing diagnostic average map sizes')
            for size in p['eval_sizes']:
                values = [aggregates[(arm, seed, mode, size)] for seed in p['seeds']]
                for key in values[0]:
                    base.close(averages[mode][arm][str(size)][key], np.mean([v[key] for v in values]),
                               f'{mode} {arm} size {size} average {key}')
    intact_rows = {(record['arm'], record['seed'], result['size']): result['episodes']
                   for record in summary['evaluation'] if record.get('mode') == 'intact'
                   for result in record['results']}
    expected_reversals = {(arm, seed, size) for arm in base.ARMS for seed in p['seeds']
                          for size in p['eval_sizes']}
    seen_reversals = set()
    for record in diagnostic['branch_reversals']:
        identity = (record['arm'], record['seed'], record['size'])
        if identity not in expected_reversals or identity in seen_reversals:
            raise ValueError('Unexpected or duplicate cue-swap branch reversal condition')
        seen_reversals.add(identity)
        arm, seed, size = identity
        left, right = intact_rows[identity], paired_rows[(arm, seed, 'cue_swapped', size)]
        both = [not a['timeout'] and not b['timeout'] for a, b in zip(left, right, strict=True)]
        reversed_pair = [valid and a['success'] != b['success']
                         for a, b, valid in zip(left, right, both, strict=True)]
        expected_values = {'assigned_pairs': len(left), 'both_reach_branch': sum(both),
                           'reversed_branch': sum(reversed_pair),
                           'both_reach_branch_fraction': np.mean(both),
                           'branch_reversal_fraction': np.mean(reversed_pair)}
        for key, value in expected_values.items():
            base.close(record[key], value, f'cue-swap {key}')
    if seen_reversals != expected_reversals:
        raise ValueError('Missing cue-swap branch reversal conditions')
    rule, size = p['memory_diagnostic'], str(p['train_size'])
    candidate = rule['candidate']
    if candidate != 'recurrent_ppo':
        raise ValueError('Expected the predeclared recurrent PPO memory-use candidate')
    means = summary['averages']
    success = means[candidate][size]['success']
    replayed_checks = {
        'same_size_success': success >= rule['same_size_success_minimum'],
        'each_fit_same_size_success': all(row['success'] >= rule['each_fit_same_size_success_minimum']
                                         for row in summary['per_fit'][candidate][size]),
        'gain_vs_current': success - means['current_ppo'][size]['success']
                           >= rule['same_size_gain_vs_current_minimum'],
        'drop_on_reset': success - means[candidate + '_reset'][size]['success']
                         >= rule['same_size_drop_on_reset_minimum'],
    }
    checks = diagnostic['memory_checks']
    if (checks != replayed_checks or any(type(value) is not bool for value in checks.values())
            or type(diagnostic['memory_gate_passed']) is not bool
            or all(checks.values()) != diagnostic['memory_gate_passed']):
        raise ValueError('Diagnostic memory gate does not replay')
    return path, diagnostic, scores


def contrast_panel(ax, scores, p):
    centers = np.arange(len(p['eval_sizes']))
    arms = base.ARMS[1:]
    width = .72 / len(arms)
    extrema = [0.]
    for index, arm in enumerate(arms):
        values = [[100 * (scores[(arm, seed, 'intact', size)]
                          - scores[(arm, seed, 'reset', size)])
                   for seed in p['seeds']] for size in p['eval_sizes']]
        x = centers + (index - (len(arms) - 1) / 2) * width
        ax.bar(x, [np.mean(v) for v in values], width * .82, color=base.COLORS[arm], alpha=.83, zorder=3)
        for position, row in zip(x, values, strict=True):
            base.mark_fits(ax, position, row, base.COLORS[arm], width=width * .5)
            extrema.extend(row)
    pad = max(8, (max(extrema) - min(extrema)) * .18)
    ax.set(xticks=centers, xticklabels=[f'{size} × {size}' for size in p['eval_sizes']],
           ylim=(max(-108, min(extrema) - pad), min(108, max(extrema) + pad)),
           ylabel='Success difference (percentage points)')
    ax.set_title('Intact minus hidden state reset each step', loc='left', fontweight='bold', pad=12)
    ax.axhline(0, color='#657185', linewidth=.8, zorder=4)
    base.format_axis(ax)


def reversal_panel(ax, diagnostic, p):
    rows = {(r['arm'], r['seed'], r['size']): r for r in diagnostic['branch_reversals']}
    centers, width = np.arange(len(p['eval_sizes'])), .18
    for index, arm in enumerate(base.ARMS):
        x = centers + (index - 1.5) * width
        values = [[100 * rows[(arm, seed, size)]['branch_reversal_fraction'] for seed in p['seeds']]
                  for size in p['eval_sizes']]
        reached = [[100 * rows[(arm, seed, size)]['both_reach_branch_fraction'] for seed in p['seeds']]
                   for size in p['eval_sizes']]
        ax.bar(x, [np.mean(v) for v in values], width * .82, color=base.COLORS[arm], alpha=.83, zorder=3)
        for position, row, both in zip(x, values, reached, strict=True):
            base.mark_fits(ax, position, row, base.COLORS[arm], width=width * .5)
            ax.scatter(position, np.mean(both), marker='_', s=75, color='#364354', linewidths=1.3, zorder=4)
    ax.set(xticks=centers, xticklabels=[f'{size} × {size}' for size in p['eval_sizes']],
           ylim=(-2, 108), yticks=np.arange(0, 101, 25), ylabel='Assigned episode pairs (%)')
    ax.set_title('Cue swap: opposite physical branch chosen', loc='left', fontweight='bold', pad=12)
    base.format_axis(ax)


def results_figure(summary, scores, diagnostic, diagnostic_scores):
    p = summary['protocol']
    fig = plt.figure(figsize=(14.5, 9.5), facecolor='white')
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=.58, wspace=.65)
    fig.subplots_adjust(left=.065, right=.98, top=.79, bottom=.17)
    fig.suptitle('Cue-visible adapted MiniGrid Memory', x=.065, y=.97,
                 ha='left', fontsize=19, fontweight='bold', color='#233044')
    fig.text(.065, .928, 'Cue visible at episode start  |  Three training fits per arm  |  Development experiment',
             fontsize=11, color='#5c6878')
    fig.legend(handles=[Patch(facecolor=base.COLORS[a], label=base.LABELS[a])
                        for a in (*base.ARMS, 'random')], loc='upper left', bbox_to_anchor=(.059, .9),
               ncol=5, frameon=False, fontsize=10, handlelength=1.1, columnspacing=1.4)
    for index, size in enumerate(p['eval_sizes']):
        ax = fig.add_subplot(grid[0, index * 2:index * 2 + 2])
        for x, arm in enumerate((*base.ARMS, 'random')):
            seeds = [None] if arm == 'random' else p['seeds']
            values = [100 * scores[(arm, seed, 'intact', size)] for seed in seeds]
            ax.bar(x, np.mean(values), .66, color=base.COLORS[arm], alpha=.83, zorder=3)
            base.mark_fits(ax, x, values, base.COLORS[arm], width=.22)
            ax.text(x, max(values) + 3, f'{np.mean(values):.1f}', ha='center', fontsize=9, color='#364354')
        shift = 'training size' if size == p['train_size'] else 'size shift'
        ax.set(xticks=range(5), xticklabels=SHORT_LABELS, ylim=(-2, 110), yticks=np.arange(0, 101, 25))
        ax.set_title(f'{size} × {size}: {shift}', loc='left', fontweight='bold', pad=12)
        if index == 0:
            ax.set_ylabel('Episodes successful (%)')
        ax.tick_params(axis='x', labelsize=9)
        base.format_axis(ax)
    contrast_panel(fig.add_subplot(grid[1, :3]), scores, p)
    reversal_panel(fig.add_subplot(grid[1, 3:]), diagnostic, p)
    fig.text(.065, .107,
             f"Bars: fit means. Circles: each fit. Random: one evaluation set. "
             f"{p['evaluation_episodes']} matched episode seeds per fit and map size.",
             fontsize=9, color='#5c6878')
    fig.text(.065, .081,
             'Cue swap preserves the reward goal. Timeouts stay in the denominator. '
             'Small horizontal marks: both runs reached a branch.', fontsize=9, color='#5c6878')
    original = 'met' if summary['continuation_passed'] else 'not met'
    memory = 'met' if diagnostic['memory_gate_passed'] else 'not met'
    fig.text(.065, .054, f'Prediction-gain gate: {original}. Memory-use gate: {memory}. '
             'No independent confirmation or novelty claim.', fontsize=9, color='#364354', fontweight='bold')
    fig.text(.065, .028, 'Predictive auxiliary heads train the policy; this experiment does not perform imagined planning.',
             fontsize=9, color='#5c6878')
    return fig


def native_figure(summary, scores, diagnostic_scores):
    p = summary['protocol']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)
    fig.subplots_adjust(left=.065, right=.98, top=.73, bottom=.22, wspace=.16)
    fig.suptitle('Transfer back to the native MiniGrid start', x=.065, y=.96,
                 ha='left', fontsize=18, fontweight='bold', color='#233044')
    fig.text(.065, .897, 'Same trained snapshots  |  No additional training  |  Matched episode seeds',
             fontsize=11, color='#5c6878')
    fig.legend(handles=[Patch(facecolor='#657185', alpha=.85, label='Cue-visible adapted start'),
                        Patch(facecolor='#657185', alpha=.3, label='Native random start')],
               loc='upper left', bbox_to_anchor=(.059, .86), ncol=2, frameon=False, fontsize=10)
    for ax, size in zip(axes, p['eval_sizes'], strict=True):
        for x, arm in enumerate(base.ARMS):
            for offset, mode, alpha in [(-.17, 'intact', .85), (.17, 'native_start', .3)]:
                mapping = scores if mode == 'intact' else diagnostic_scores
                values = [100 * mapping[(arm, seed, mode, size)] for seed in p['seeds']]
                ax.bar(x + offset, np.mean(values), .29, color=base.COLORS[arm], alpha=alpha, zorder=3)
                base.mark_fits(ax, x + offset, values, base.COLORS[arm], width=.1)
            for seed, jitter in zip(p['seeds'], np.linspace(-.05, .05, len(p['seeds'])), strict=True):
                ax.plot([x - .17 + jitter, x + .17 + jitter],
                        [100 * scores[(arm, seed, 'intact', size)],
                         100 * diagnostic_scores[(arm, seed, 'native_start', size)]],
                        color=base.COLORS[arm], alpha=.35, linewidth=.8, zorder=4)
        ax.set(xticks=range(4), xticklabels=SHORT_LABELS[:4], ylim=(-2, 110), yticks=np.arange(0, 101, 25))
        ax.set_title(f'{size} × {size}', loc='left', fontweight='bold', pad=12)
        base.format_axis(ax)
    axes[0].set_ylabel('Episodes successful (%)')
    fig.text(.065, .11, 'Bars: fit means. Circles and connecting lines: paired training fits. '
             'The native start changes the information-acquisition task.', fontsize=9, color='#5c6878')
    fig.text(.065, .065, 'Cue-visible success alone does not establish transfer to the original task. '
             'Development diagnostic, not an independent confirmation.', fontsize=9, color='#364354')
    return fig


def publish(args):
    if not args.report.is_dir():
        raise ValueError('--report must be a directory containing summary.json, plan.json and diagnostic.json')
    completion_path = args.report / 'report-completed.json'
    completion = json.loads(completion_path.read_text())
    expected_completion = {
        'status': 'completed', 'protocol_version': VERSION,
        'plan_sha256': base.sha(args.report / 'plan.json'),
        'summary_sha256': base.sha(args.report / 'summary.json'),
        'diagnostic_sha256': base.sha(args.report / 'diagnostic.json'),
    }
    if completion != expected_completion:
        raise ValueError('A complete report with matching plan, summary and diagnostics is required')
    source, plan_path, summary, training, scores = base.load_completed_report(args.report)
    if summary['protocol']['version'] != VERSION:
        raise ValueError('This publisher only accepts the cue-memory-v1 adapted-start experiment')
    diagnostic_path, diagnostic, diagnostic_scores = load_diagnostics(
        args.report, source, plan_path, summary, training)
    interactions, curves, learning_hashes = base.load_learning_curves(args.run, summary, training)
    for (arm, seed), receipt in training.items():
        fit = args.run / 'fits' / f'{arm}-{seed}'
        if (json.loads((fit / 'completed.json').read_text()) != receipt
                or base.sha(fit / 'model.pt') != receipt['checkpoint_sha256']):
            raise ValueError('Published summary does not match preserved trained snapshot and receipt')
    expected = [f'{stem}.{suffix}' for stem in STEMS for suffix in ('png', 'svg')] + [RECEIPT]
    if any((args.out / name).exists() for name in expected):
        raise FileExistsError('Figure export already exists; use a fresh output directory')
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
             'svg.hashsalt': VERSION, 'savefig.facecolor': 'white'}
    rendered, figures = {}, []
    with plt.rc_context(style):
        try:
            figures.append((STEMS[0], results_figure(summary, scores, diagnostic, diagnostic_scores)))
            learning = base.learning_figure(summary, interactions, curves)
            learning.suptitle('Training trajectories: cue-visible adapted MiniGrid', x=.09, y=.977,
                              ha='left', fontsize=17, fontweight='bold', color='#233044')
            figures.append((STEMS[1], learning))
            figures.append((STEMS[2], native_figure(summary, scores, diagnostic_scores)))
            for stem, fig in figures:
                for suffix in ('png', 'svg'):
                    buffer = io.BytesIO()
                    metadata = {'Date': None, 'Creator': 'OpenJev cue-memory-v1 publisher'} if suffix == 'svg' else {}
                    fig.savefig(buffer, format=suffix, dpi=180, bbox_inches='tight', pad_inches=.18,
                                metadata=metadata)
                    content = buffer.getvalue()
                    if suffix == 'svg':
                        content = ('\n'.join(line.rstrip() for line in content.decode().splitlines()) + '\n').encode()
                    rendered[f'{stem}.{suffix}'] = content
        finally:
            for _, fig in figures:
                plt.close(fig)
    receipt = {
        'protocol_version': VERSION, 'summary_sha256': base.sha(source), 'plan_sha256': base.sha(plan_path),
        'report_completed_sha256': base.sha(completion_path),
        'diagnostic_sha256': base.sha(diagnostic_path), 'publisher_sha256': base.sha(Path(__file__)),
        'base_publisher_sha256': base.sha(Path(base.__file__)),
        'figures': {name: hashlib.sha256(content).hexdigest() for name, content in rendered.items()},
        'learning_sources_sha256': learning_hashes,
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
        'training_fits': len(training), 'training_seeds': summary['protocol']['seeds'],
        'evaluation_sizes': summary['protocol']['eval_sizes'],
        'diagnostic_conditions': len(diagnostic['evaluations']),
        'scope': 'Cue-visible adapted MiniGrid development experiment; all fits; no confidence intervals or novelty claim',
        'learning_scope': 'Rolling last 100 completed training episodes; not held-out evaluation',
        'native_scope': 'Same snapshots returned to the native information-acquisition start; no retraining',
        'deterministic_rendering': 'Fixed SVG hash salt, no SVG date, trailing SVG whitespace removed before hashing',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        with (args.out / name).open('xb') as stream:
            stream.write(content)
    with (args.out / RECEIPT).open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Completed cue-visible report directory')
    parser.add_argument('--run', type=Path, required=True, help='Core execution directory containing fits/')
    parser.add_argument('--out', type=Path, required=True, help='Fresh figure export directory')
    publish(parser.parse_args())
